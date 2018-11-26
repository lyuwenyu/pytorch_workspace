import sys
import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
# from tensorboardX import SummaryWriter

filedir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(filedir, '..', 'utils')
sys.path.insert(0, path)
sys.path.insert(0, filedir)
from utils.parse_config import parse_config
from build_target import build_target, _build_target
from utils.ops_logger import writer

from build_textbox_layer import textbox_layer
from build_yolo_layer import YOLOLayer
from build_ssd_layer import SSDLayer

VERSION = torch.__version__

def create_model(model_defs, cls_num, img_dim=None, quad=False, yolo=False, ssd=False):
    '''
    '''
    hyperparameter = model_defs.pop(0)
    outfilters = [int(hyperparameter['channels'])]
    module_list = nn.ModuleList()
    
    # print(model_defs)

    for i, defs in enumerate(model_defs):
        module = nn.Sequential()
        # print(defs)

        if 'convolutional' in defs['type']:

            bn = int(defs.get('batch_normalize', 0))
            filters = int(defs['filters'])
            kernel_size = int(defs['size'])
            pad = (kernel_size - 1) // 2 if int(defs['pad']) else 0
            stride = int(defs['stride'])

            module.add_module(f'conv_{i}', nn.Conv2d(in_channels=outfilters[-1],
                                                     out_channels=filters,
                                                     kernel_size=kernel_size,
                                                     stride=stride,
                                                     padding=pad,
                                                     bias=not bn))   
            if bn:
                module.add_module(f'bn_{i}', nn.BatchNorm2d(filters))
            
            if defs['activation'] == 'leaky':
                module.add_module(f'leaky_{i}', nn.LeakyReLU(0.1))

        elif defs['type'] == 'upsample':
            if VERSION >= '0.4.1':
                module.add_module(f'upsample_{i}', EmptyLayer())
            else:
                upsample = nn.Upsample(scale_factor=int(defs['stride']), mode='nearest')
                module.add_module(f'upsample_{i}', upsample)
        
        elif 'route' in defs['type']:
            layers = [int(l) for l in defs['layers'].split(',')]
            filters = sum(outfilters[l] for l in layers)
            module.add_module(f'route_{i}', EmptyLayer())

        elif 'shortcut' in defs['type']:
            filters = outfilters[int(defs['from'])]
            module.add_module(f'shortcut_{i}', EmptyLayer())

        elif defs['type'] == 'yolo':
            anchor_index = [int(l) for l in defs['mask'].split(',')]
            anchors = [float(x) for x in defs['anchors'].split(',')]
            anchors = [(anchors[j], anchors[j+1]) for j in range(0, len(anchors), 2)]
            anchors = [anchors[j] for j in anchor_index]

            if cls_num is not None:
                num_classes = cls_num
            else:
                num_classes = int(defs['classes'])

            if quad:
                yolo_layer = textbox_layer(anchors, num_classes, anchor_index, img_dim)
                attr_num = 4 + 8 + 1 + num_classes
                # print(attr_num)

            # elif ssd:
            #     yolo_layer = SSDLayer(num_classes)

            else:
                yolo_layer = YOLOLayer(anchors, num_classes, anchor_index, img_dim)
                attr_num = 4 + 1 + num_classes

            module.add_module(f'yolo_{i}', yolo_layer)

            # 
            for n, m in module_list[-1].named_children():
                name = n
                kernel_size = m.kernel_size
                stride = m.stride
                in_channels = m.in_channels
            
            _m = nn.Conv2d(in_channels, attr_num * len(anchors), kernel_size, stride)
            print(_m)
            del module_list[-1] # __delitem__
            module_list += [nn.Sequential(OrderedDict([(name, _m)]))] # __iadd__ append
            # module_list[len(module_list) - 1] = nn.Sequential(OrderedDict([(name, _m)]))
            # `[-1]` makes wrong, using `len(module_list) - 1` instead. when __setitem__

        outfilters += [filters]
        module_list += [module]

    return module_list


#---
class EmptyLayer(nn.Module):
    def __init__(self, ):
        super(EmptyLayer, self).__init__()


#---
class DarkNet(nn.Module):
    def __init__(self, cfg, cls_num, quad=False):
        super(DarkNet, self).__init__()
        
        self.module_defs = parse_config(cfg)
        # self.module_defs[0]['height'] = img_size
        self.module_list = create_model(self.module_defs, cls_num=cls_num, quad=quad)

        self.loss_names = ['loss', 'x', 'y', 'w', 'h', 'conf', 'cls' ]
        
    def forward(self, x, target=None, requestPrecision=False, epoch=None):
        
        layer_outputs = []
        outputs = []
        losses = []
        # losses = dict.fromkeys(self.loss_names, 0)
        
        # output4paralel = np.zeros((1, len(self.loss_names)))
        tic = time.time()

        for _, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):

            if module_def['type'] == 'convolutional':
                x = module(x)

            elif module_def['type'] == 'route':
                layers = [int(l) for l in module_def['layers'].split(',')]
                x = torch.cat([layer_outputs[l] for l in layers], dim=1)

            elif module_def['type'] == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]

            elif module_def['type'] == 'upsample':
                if VERSION == '0.4.1':
                    x = F.interpolate(layer_outputs[-1], scale_factor=int(module_def['stride']), mode='nearest',)
                else:
                    x = module(x)

            elif module_def['type'] == 'yolo':
                if target is None: # test phase)
                    x = module(x)
                    outputs += [x]

                else: # training phase
                    x = module[0](x, target=self.set_target(target))  # module is sequential object, not yolo
                    outputs += [x[0]]
                    # for ii, (ni, xi) in enumerate(zip(self.loss_names, x)):
                    #     output4paralel[0, ii] += [xi.item()]
                    #     losses[ni] += xi.item()
                    losses += [[xi.item() for xi in x]]
            
            layer_outputs += [x]
            # print(time.time() - tic)

        # print(time.time() - tic)
        if target is None:
            return torch.cat(outputs, dim=1)

        else:
            return sum(outputs)
            # return sum(outputs), losses # quad


    def set_target(self, target):
        ''' get valid target'''
        bboxes = []
        for ii in range(target.shape[0]):
            _target = target[ii]
            _target = _target[_target[:, 0] == 1]
            # if _target.shape[0] == 0:
            #     continue
            bboxes += [_target[:, 1:]]
        return bboxes


    def load_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        fp = open(weights_path, 'rb')
        header = np.fromfile(fp, dtype=np.int32, count=5)  # First five are header values

        # Needed to write header when saving weights
        self.header_info = header

        self.seen = header[3]
        weights = np.fromfile(fp, dtype=np.float32)  # The rest are weights
        fp.close()

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def['type'] == 'convolutional':
                conv_layer = module[0]
                if module_def.get('batch_normalize', 0):
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w




if __name__ == '__main__':
    path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(path, '..', '_model', 'yolov3_v1.cfg')
    
    device = torch.device('cpu')
    data = torch.rand(1, 3, 256, 320).to(device=device)
    model = DarkNet(path, cls_num=10, quad=False).to(device=device)
    print(model)
    # model.load_state_dict(torch.load(os.path.join(filedir, 'yolov3.torch')))
    # model.eval()
    # model = model.cuda()

    # model.load_weights(weights_path=os.path.join('/home/wenyu/workspace/pytorch_workspace/yolov3/_model', 'yolov3.weights'))
    # torch.save(model.state_dict(), os.path.join(filedir, 'yolov3.torch'))

    times = []
    for _ in range(10):
        tic = time.time()
        model(data)
        times += [time.time() - tic]

    print(sum(times) / len(times))

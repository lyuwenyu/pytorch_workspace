import sys
import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

filedir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(filedir, '..', 'utils')
sys.path.insert(0, path)
sys.path.insert(0, filedir)
from utils.parse_config import parse_config
from build_target import build_target


VERSION = torch.__version__

def create_model(model_defs):
    '''
    '''
    hyperparameter = model_defs.pop(0)
    outfilters = [int(hyperparameter['channels'])]
    module_list = nn.ModuleList()

    for i, defs in enumerate(model_defs):
        module = nn.Sequential()

        if defs['type'] == 'convolutional':

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
        
        elif defs['type'] == 'route':
            layers = [int(l) for l in defs['layers'].split(',')]
            filters = sum(outfilters[l] for l in layers)
            module.add_module(f'route_{i}', EmptyLayer())

        elif defs['type'] == 'shortcut':
            filters = outfilters[int(defs['from'])]
            module.add_module(f'shortcut_{i}', EmptyLayer())

        elif defs['type'] == 'yolo':
            anchor_index = [int(l) for l in defs['mask'].split(',')]
            anchors = [float(x) for x in defs['anchors'].split(',')]
            anchors = [(anchors[j], anchors[j+1]) for j in range(0, len(anchors), 2)]
            anchors = [anchors[j] for j in anchor_index]

            num_classes = int(defs['classes'])
            img_height = int(hyperparameter['height'])

            yolo_layer = YOLOLayer(anchors, num_classes, img_height, anchor_index)
            module.add_module(f'yolo_{i}', yolo_layer)

        outfilters += [filters]
        module_list += [module]

    return module_list


#---
class EmptyLayer(nn.Module):
    def __init__(self, ):
        super(EmptyLayer, self).__init__()

#---
class YOLOLayer(nn.Module):
    def __init__(self, anchors, num_classes, img_height, anchor_index):
        super(YOLOLayer, self).__init__()
        
        self.anchors = anchors
        self.nA = len(anchors)
        self.nC = num_classes
        self.bbox_attrs = num_classes + 5
        self.img_dim = img_height

        if anchor_index[0] == 6:
            stride = 32
        elif anchor_index[0] == 3:
            stride = 16
        else:
            stride = 8

        self.nG = int(img_height / stride)
        self.stride = stride

        self.scaled_anchors = (torch.tensor(anchors)/stride).to(torch.float32)        
        self.anchor_w = self.scaled_anchors[:, 0]
        self.anchor_h = self.scaled_anchors[:, 1]

        if VERSION >= '0.4.1':
            self.grid_x, self.grid_y = torch.meshgrid((torch.arange(self.nG), torch.arange(self.nG)))
        else:
            self.grid_x = torch.arange(self.nG).repeat(self.nG, 1)
            self.grid_y = torch.arange(self.nG).repeat(self.nG, 1).t()

        print(self.scaled_anchors.shape)

        self.mseLoss = nn.MSELoss(size_average=True)
        # self.crossEntropyLoss = nn.CrossEntropyLoss(size_average=True)
        self.bceLoss = nn.BCELoss(size_average=True)
        # self.bceLoss2 = nn.BCELoss(size_average=False)

    def forward(self, p, target=None, requestPrecision=False, epoch=None):
        '''forward '''
        bs = p.shape[0]
        nG = p.shape[2]
        stride = self.stride
        assert nG == self.nG, 'grid pixel should be same.'
        
        p = p.view(bs, self.nA, self.bbox_attrs, nG, nG).permute(0, 1, 3, 4, 2).contiguous()

        x = torch.sigmoid(p[..., 0])
        y = torch.sigmoid(p[..., 1])
        w = p[..., 2]
        h = p[..., 3]

        width = torch.exp(w) * self.anchor_w.view(1, -1, 1, 1).to(dtype=w.dtype, device=w.device) #.data
        height = torch.exp(h) * self.anchor_h.view(1, -1, 1, 1).to(dtype=h.dtype, device=h.device)

        pred_boxes = torch.zeros((bs, self.nA, nG, nG, 4)).to(dtype=p.dtype, device=p.device)
        pred_conf = p[..., 4]
        pred_cls = p[..., 5:]

        if target is None: # inference phase
            pred_boxes[..., 0] = x + self.grid_x.to(dtype=x.dtype, device=x.device)
            pred_boxes[..., 1] = y + self.grid_y.to(dtype=y.dtype, device=y.device)
            pred_boxes[..., 2] = width
            pred_boxes[..., 3] = height

            out = torch.cat((
                pred_boxes.view(bs, -1, 4) * stride,
                torch.sigmoid(pred_conf).view(bs, -1, 1),
                pred_cls.view(bs, -1, self.nC)),
                dim=-1)

            return out
        
        else: # train phase TODO 

            if requestPrecision:
                pred_boxes[..., 0] = x + self.grid_x.to(dtype=x.dtype, device=x.device) - width / 2
                pred_boxes[..., 1] = y + self.grid_y.to(dtype=x.dtype, device=x.device) - height / 2
                pred_boxes[..., 2] = x + self.grid_x.to(dtype=x.dtype, device=x.device) + width / 2
                pred_boxes[..., 3] = x + self.grid_y.to(dtype=x.dtype, device=x.device) + height / 2

            tx, ty, tw, th, tconf, tcls, conf_mask = build_target(pred_boxes, 
                                                            pred_conf, 
                                                            pred_cls, 
                                                            target, 
                                                            self.scaled_anchors, 
                                                            self.nA, self.nC, self.nG,
                                                            requestPrecision)
            nM = tconf.sum().float()
            mask = tconf

            if nM > 0:
                lx = self.mseLoss(x[mask], tx[mask])
                ly = self.mseLoss(y[mask], ty[mask])
                lw = self.mseLoss(w[mask], tw[mask])
                lh = self.mseLoss(h[mask], th[mask])
                lcls = self.bceLoss(pred_cls[mask], tcls[mask])
                # lconf = self.bceLoss2(pred_conf[tconf], tconf[tconf].to(dtype=pred_conf.dtype, device=pred_conf.device))
                lconf = self.bceLoss(pred_conf[conf_mask], tconf[conf_mask].to(dtype=pred_conf.dtype, device=pred_conf.device))
                # lcls = self.crossEntropyLoss(pred_cls[tconf], torch.argmax(tcls[tconf], 1))
                
            else:
                lx, ly, lw, lh, lcls, lconf = [torch.tensor(0.).to(dtype=torch.float32, )]*6

            # lconf += nM * self.bceLoss1(pred_conf[~tconf], tconf[~tconf].to(dtype=pred_conf.dtype, device=pred_conf.device))
            # lconf = lconf / nM

            loss = lx + ly + lw + lh + lconf + lcls

        return loss 


#---
class DarkNet(nn.Module):
    def __init__(self, cfg, img_size):
        super(DarkNet, self).__init__()
        
        self.module_defs = parse_config(cfg)
        self.module_defs[0]['height'] = img_size
        self.module_list = create_model(self.module_defs)

        self.loss_names = ['loss', 'x', 'y', 'w', 'h', 'conf', 'cls', ]

    def forward(self, x, target=None, requestPrecision=False, epoch=None):
        
        layer_outputs = []
        outputs = []

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

                else: # training phase TODO
                    pass
                
                outputs += [x]
            
            layer_outputs += [x]


        if target is None:
            return torch.cat(outputs, dim=1)

        else:
            pass


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
    path = os.path.join(path, '..', '_model', 'yolov3.cfg')
    
    data = torch.rand(2, 3, 416, 416).to(device=torch.device('cuda'))
    model = DarkNet(path, img_size=416)
    model.load_state_dict(torch.load(os.path.join(filedir, 'yolov3.torch')))
    model.eval()
    model = model.cuda()

    # model.load_weights(weights_path=os.path.join(filedir, 'yolov3.weights'))
    torch.save(model.state_dict(), os.path.join(filedir, 'yolov3.torch'))



    tic = time.time()
    print(model(data).shape)
    print(time.time() - tic)
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable ## 

import numpy as np
from collections import OrderedDict
from PIL import ImageDraw, Image

from util import predict_transform, get_results, load_classes, pre_image

def parse_cfg(cfgfile):

    with open(cfgfile, 'r') as f:
        
        lines = f.readlines()
        lines = [l for l in lines if len(l)>0]
        lines = [l.strip() for l in lines if l[0] !='#']
        lines = [l for l in lines if l!='']

    block = {}
    blocks = []

    for line in lines:

        if line[0] == '[':
            if len(block) != 0:
                blocks += [block]
                block = {}
            block['type'] = line[1:-1].strip()
        else:
            k, v = line.split('=')
            block[k.strip()] = v.strip()

    blocks += [block]
    
    return blocks


class EmptyLayer(nn.Module):
    def __init__(self, ):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


def create_modules(blocks):
    
    net_info = blocks[0]
    
    module_list = nn.ModuleList()

    pre_filters = 3
    out_filters = []

    for index, m in enumerate(blocks[1:]):
        
        module = nn.Sequential()

        if m['type'] == 'convolutional':
            
            activation = m['activation']
            filters = int(m['filters'])
            padding = int(m['pad'])
            kernel_size = int(m['size'])
            stride = int(m['stride'])

            if padding:
                pad = (kernel_size-1) // 2
            else:
                pad = 0

            try:
                bn = int(m['batch_normalize'])
                bias = False
            except:
                bn = False
                bias = True

            conv = nn.Conv2d(pre_filters, out_channels=filters, kernel_size=kernel_size, stride=stride, padding=pad, bias=bias)
            module.add_module('conv_{}'.format(index), conv)

            if bn:
                module.add_module('batch_norm_{}'.format(index), nn.BatchNorm2d(filters))

            if activation == 'leaky':
                module.add_module('leaky_{}'.format(index), nn.LeakyReLU(0.1, inplace=True))

        ## 
        elif m['type'] == 'upsample':

            stride = m['stride']
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            module.add_module('upsample_{}'.format(index), upsample)

        ##
        elif m['type'] == 'shortcut':

            module.add_module('shortcut_{}'.format(index), EmptyLayer())

        ## 
        elif m['type'] == 'route':
            
            m['layers'] = [x.strip() for x in m['layers'].split(',')]
            start = int(m['layers'][0])

            try:
                end = int(m['layers'][1])
            except:
                end = 0

            if start > 0:
                start = start - index

            if end > 0:
                end = end - index

            module.add_module('route_{}'.format(index), EmptyLayer())

            if end < 0:
                filters = out_filters[index+start] + out_filters[index+end]
            else:
                filters = out_filters[index+start]

        ## 
        elif m['type'] == 'yolo':
            
            mask = m['mask'].split(',')
            mask = [ int(x.strip()) for x in mask]

            anchors = m['anchors'].split(',')
            anchors = [int(x.strip()) for x in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2) ]
            anchors = [anchors[i] for i in mask]

            module.add_module('detection_{}'.format(index), DetectionLayer(anchors))

        else:
            print('--------------------------------')
            print(m['type'])
            print('--------------------------------')

        module_list += [module]
        pre_filters = filters
        out_filters += [filters]
    
    # print(out_filters)

    return net_info, module_list



class DarkNet(nn.Module):
    def __init__(self, cfgfile):
        super(DarkNet, self).__init__()

        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)


    def forward(self, x):
        
        modules = self.blocks[1:]
        outputs = {}
        
        detections = []

        for i, m in enumerate(modules):

            mtype = m['type']
            # print(m)
            # print(mtype)
            # print(x.size())

            if mtype == 'convolutional' or mtype == 'upsample':
                x = self.module_list[i](x)

            elif mtype == 'route':
                layers = m['layers']
                layers = [int(x) for x in layers]
                
                if layers[0] > 0:
                    layers[0] -= i

                if len(layers) == 1:
                    x = outputs[i+layers[0]]
                else:
                    if layers[1] > 0:
                        layers[1] -= i

                    x = torch.cat((outputs[layers[0]+i], outputs[layers[1]+i]), 1)

            elif mtype == 'shortcut':
                
                _from = int(m['from'])
                x = outputs[i-1] + outputs[i+_from]
                
            elif mtype == 'yolo':
                anchors = self.module_list[i][0].anchors
                inp_dim = int(self.net_info['height'])
                num_classes = int(m['classes'])

                detections += [ predict_transform(x, inp_dim, anchors, num_classes) ]

            outputs[i] = x

        res = torch.cat(detections, dim=1)
        # res = detections[0]
        # for dets in detections[1:]:
        #     res = torch.cat([res, dets], dim=1)
        return res

    def load_weights(self, weightfile):
        #Open the weights file
        fp = open(weightfile, "rb")
    
        #The first 5 values are header information 
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number 
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]   
        
        weights = np.fromfile(fp, dtype = np.float32)
        
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]
    
            #If module_type is convolutional load weights
            #Otherwise ignore.
            
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0
            
                conv = model[0]
                
                
                if (batch_normalize):
                    bn = model[1]
        
                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()
        
                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
        
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    #Cast the loaded weights into dims of model weights. 
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
        
                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                
                else:
                    #Number of biases
                    num_biases = conv.bias.numel()
                
                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                
                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)
                
                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)
                    
                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()
                
                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights
                
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)



if __name__ == '__main__':

    # blocks = parse_cfg('yolov3.cfg')

    # info, modules = create_modules(blocks)


    darknet = DarkNet('yolov3.cfg')
    darknet.load_weights('yolov3.weights')
    classes = load_classes('coco.names')

    img = pre_image('dog-cycle-car.png', 416)
    img = Variable(img)
    
    detections = darknet(img)

    result = get_results(detections.data)
    
    result = torch.clamp(result, 0., float(416))
    result = np.array(result)

    origin_image = Image.open('dog-cycle-car.png')
    draw = ImageDraw.Draw(origin_image)
    
    for i in range(result.shape[0]):
        res = result[i]
        draw.rectangle( (res[1], res[2], res[3], res[4]), outline=(255, 0, 0))

    origin_image.save('out.jpg')
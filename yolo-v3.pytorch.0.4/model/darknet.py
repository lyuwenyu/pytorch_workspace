import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable ## 

import numpy as np
from collections import OrderedDict


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
    
    print(out_filters)

    return net_info, module_list



class DarkNet(nn.Module):
    def __init__(self, cfgfile):
        super(DarkNet, self).__init__()

        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)


    def forward(self, x):
        
        modules = self.blocks[1:]
        outputs = {}
        write = 0

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

                # x = predict_transform(x, inp_dim, anchors, num_classes)
                # if not write:
                #     detections = x
                #     write = 1
                # else:
                #     detections  = torch.cat((detections, x), 1)

            outputs[i] = x

        return 0



if __name__ == '__main__':

    # blocks = parse_cfg('yolov3.cfg')

    # info, modules = create_modules(blocks)

    # print(info)
    # print(modules)
    # print(blocks)


    darknet = DarkNet('yolov3.cfg')

    # x = Variable(torch.randn(1, 3, 512, 512))
    x = torch.randn(1, 3, 512, 512)
    print(darknet)
    
    print( darknet(x) )


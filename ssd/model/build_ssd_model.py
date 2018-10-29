import torch
import torch.nn as nn
import torchvision.models as models

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_ssd_layer import SSDLayer


class SSD(nn.Module):
    def __init__(self, num_classes):
        super(SSD, self).__init__()

        self.basenet = models.resnet50(pretrained=True)
        self.basenet = nn.ModuleList(self.basenet.children())[:-2]
        self.attr_num = num_classes + 1 + 4 
        self.encodes = nn.ModuleList([nn.Conv2d(c, self.attr_num * s, 1, 1) for c, s in zip([512, 1024, 2048], [4, 4, 4])]) # for resnet50
        # self.encodes = nn.ModuleList([nn.Conv2d(c, self.attr_num * s, 1, 1) for c, s in zip([128, 256, 512], [5, 7, 7])])
        self.ssdlayer = SSDLayer()
     
    def forward(self, x, target=None):
        
        features = []

        for m in self.basenet[:-3]:
            x = m(x)
        
        for i, m in enumerate(self.basenet[-3:]):
            x = m(x)
            # print(x.shape)
            out = self.encodes[i](x)
            out = out.view(out.shape[0], -1, self.attr_num, out.shape[2], out.shape[3]).permute(0, 1, 3, 4, 2).contiguous()
            out = out.view(out.shape[0], -1, self.attr_num).contiguous()
            features += [out]
        
        features = torch.cat(features, dim=1)
        # print('features: ', features.shape)
        out = self.ssdlayer(features, target)

        return out



if __name__ == '__main__':
    

    data = torch.rand(2, 3, 512, 512)
    target = [torch.rand(7, 5)] * 2

    ssd = SSD(8)

    ssd(data, target)
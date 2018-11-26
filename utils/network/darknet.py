import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import random
import numpy as np 


class basic_block(nn.Module):
    def __init__(self, c_in, c_out, shotcut=True):
        super(basic_block, self).__init__()

        self.shotcut = shotcut
        self.mm = nn.Sequential(
            nn.Conv2d(c_in, c_out, 1, 1, padding=0),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU(0.1),

            nn.Conv2d(c_out, c_out * 2, 3, 1, padding=1),
            nn.BatchNorm2d(c_out * 2),
            nn.LeakyReLU(0.1),
        )
        
    def forward(self, x):
        '''
        x: n c h w
        '''
        if self.shotcut:
            return x + self.mm(x)
        else:
            return self.mm(x)
# ----


class DarkNet(nn.Module):
    def __init__(self, filters=(64, 128, 256, 512, 1024), blocks=(1, 3, 8, 8, 5)):
        super(DarkNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 1, 1, padding=0),
            nn.LeakyReLU(0.1),
        )

        self.blocks = nn.ModuleList([nn.Sequential(
            nn.Conv2d(c//2, c, 3, 2, padding=1), # downsample
            *[basic_block(c, c//2)] * n) for c, n in zip(filters, blocks)
        ])

    def forward(self, x):
        '''
        x: n c h w
        '''
        x = self.conv1(x)
        
        features = []
        for block in self.blocks:
            x = block(x)
            features += [x]

        return features


class Yolov3(nn.Module):
    def __init__(self, num_classes=80, blocks=(1, 3, 8, 8, 5), heads=(3, 3, 3)):
        super(Yolov3, self).__init__()

        # self.num_classes = num_classes
        num_attrs = (num_classes + 1 + 4) * 3
        base_filters=(64, 128, 256, 512, 1024)
        head_filters = (1024, 512, 256)
        head_pre = (256, 128)
        head_ins = (1024, 512 + 256, 256 + 128)
        
        self.darknet = DarkNet(filters=base_filters, blocks=blocks)

        self.pre_heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(inc, prec, 1, 1), nn.BatchNorm2d(prec), nn.LeakyReLU(0.1))
            for inc, prec in zip(head_filters, head_pre)])

        self.heads = nn.ModuleList()
        for cin, cout, n in zip(head_ins, head_filters, heads):
            self.heads += [nn.Sequential(
                basic_block(cin, cout//2, shotcut=False),
                *[basic_block(cout, cout//2, shotcut=False)] * (n-1)
            )]

        self.encode = nn.ModuleList([nn.Conv2d(c, num_attrs, 1, 1, ) for c in head_filters])

    def forward(self, x):
        '''
        '''
        *feas, out = self.darknet(x)

        codes = []
        
        for i, head in enumerate(self.heads):

            if i > 0:
                out = self.pre_heads[i-1](out)
                out = F.interpolate(out, scale_factor=2, mode='nearest')
                out = torch.cat([out, feas[-i]], dim=1)

            out = head(out)
            code = self.encode[i](out)
            codes += [code]
            # print(out.shape)
            # print(code.shape)

        return codes


def yolov3():
    return Yolov3(blocks=(1, 3, 8, 8, 4), heads=(3, 3, 3))


if __name__ == '__main__':

    # darknet = DarkNet(blocks=(1, 3, 8, 8, 5))
    # print(darknet)
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

    yolov3 = Yolov3(blocks=(1, 3, 5, 5, 3), heads=(3, 3, 3)).to(device)
    print(yolov3)

    x  = torch.rand(1, 3, 256, 320).to(device)

    times = []
    for _ in range(50):
        tic = time.time()
        out = yolov3(x)
        times += [time.time() - tic]
    
    print(np.mean(times))
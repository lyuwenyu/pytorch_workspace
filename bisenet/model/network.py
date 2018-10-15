import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import time

class ARM(nn.Module):
    def __init__(self, in_channels):
        super(ARM, self).__init__()

        self.m = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        '''
        forward
        '''
        # w = self.m(x)
        return self.m(x) * x


class FFM(nn.Module):
    def __init__(self, channel_1, channel_2):
        super(FFM, self).__init__()

        channels = channel_1 + channel_2

        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1),
            nn.Sigmoid(),
        )
    
    def forward(self, feat1, feat2):
        '''
        '''
        feat = torch.cat([feat1, feat2], dim=1)
        feat = self.conv(feat)
        w = self.attention(feat)

        return w * feat + feat


class SpatialNet(nn.Module):
    def __init__(self, ):
        super(SpatialNet, self).__init__()

        self.spatial_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
    
    def forward(self, x):
        '''
        '''
        return self.spatial_net(x)


class ContextNet(nn.Module):
    def __init__(self, ):
        super(ContextNet, self).__init__()
        basenet = models.resnet18(pretrained=True)
        self.basenet = nn.ModuleList(list(basenet.children())[:-2])
        
        self.att_16 = ARM(256)
        self.att_32 = ARM(512)
        self.global_ave_pooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        '''
        '''
        # img_dim = x.shape[-1]
        # outs = [m(x) for m in self.basenet]
        # out_feas = []

        tic = time.time()
        outs = []
        for _, m in enumerate(self.basenet):
            x = m(x)
            outs += [x]
        print('resnet18: ', time.time() - tic)

        fea_att_16 = self.att_16(outs[-2])
        fea_att_32 = self.att_32(outs[-1])
        fea_global = self.global_ave_pooling(outs[-1])

        fea_att_32 = fea_att_32 * fea_global
        fea_att_32 = F.interpolate(fea_att_32, scale_factor=4, mode='bilinear')
        fea_att_16 = F.interpolate(fea_att_16, scale_factor=2, mode='bilinear')

        out = torch.cat([fea_att_16, fea_att_32], dim=1)

        return out


class BiSeNet(nn.Module):
    def __init__(self, num_classes):
        super(BiSeNet, self).__init__()

        self.spatial = SpatialNet()
        self.context = ContextNet()
        self.ffm = FFM(256, 256 + 512)
        self.conv = nn.Conv2d(256+256+512, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        '''
        '''
        tic = time.time()
        out_spatial = self.spatial(x)
        print('spatial: ', time.time() - tic)

        tic = time.time()
        out_context = self.context(x)
        print('context: ', time.time() - tic)

        tic = time.time()
        feat = self.ffm(out_spatial, out_context)
        print('ffm: ', time.time() - tic)

        feat = F.interpolate(feat, scale_factor=8, mode='bilinear')
        logits = self.conv(feat)

        return logits



if __name__ == '__main__':

    device = torch.device('cuda:0')

    # m = models.resnet18(pretrained=True)
    data = torch.rand(2, 3, 512, 512)



    # arm = ARM(10)
    # out = arm(data)
    # print(out.size())

    # net = ContextNet()
    # out = net(data)
    # print(out.shape)

    net = BiSeNet(10)
    net.to(device=device)
    data = data.to(device=device) # model.to() is difference from data.to()
    
    # print(net)

    tic = time.time()
    out = net(data)
    print(time.time() - tic)

    print(out.shape)

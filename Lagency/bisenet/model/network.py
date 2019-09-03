import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import gluoncvth
import time



class Conv_Norm_ReLU(nn.Module):
    def __init__(self, in_c, out_c, kernel=3, stride=2, padding=1, relu=True, group_norm=False):
        super(Conv_Norm_ReLU, self).__init__()

        self.conv = nn.Conv2d(in_c, out_c, kernel, stride, padding)

        if group_norm:
            self.norm = nn.GroupNorm(num_groups = out_c // 16, num_channels=out_c)
        else:
            self.norm = nn.BatchNorm2d(out_c),

        self.relu = relu

    def forward(self, x):

        x = self.conv(x)
        x = self.norm(x)
        if self.relu:
            x = F.relu(x)
        return x


class ARM(nn.Module):
    '''
    attention refine module
    '''
    def __init__(self, in_channels):
        super(ARM, self).__init__()

        self.m = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1),
            # nn.BatchNorm2d(in_channels),
            nn.GroupNorm(num_groups=in_channels//16, num_channels=in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        '''
        forward
        '''
        w = self.m(x)
        return w * x


class FFM(nn.Module):
    '''
    feature fusion module
    '''
    def __init__(self, channel_1, channel_2):
        super(FFM, self).__init__()

        channels = channel_1 + channel_2

        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(channels),
            nn.GroupNorm(num_groups=channels//16, num_channels=channels),
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
            # nn.BatchNorm2d(64),
            nn.GroupNorm(num_groups=64//16, num_channels=64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(128),
            nn.GroupNorm(num_groups=128//16, num_channels=128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(256),
            nn.GroupNorm(num_groups=256//16, num_channels=256),
            nn.ReLU(),
        )
    
    
    def forward(self, x):
        '''
        '''
        return self.spatial_net(x)


class ContextNet(nn.Module):
    def __init__(self, ):
        super(ContextNet, self).__init__()
        self.basenet = models.resnet18(pretrained=True)
        # self.basenet = gluoncvth.models.resnet18(pretrained=True)
        self.basenet = nn.ModuleList(list(self.basenet.children())[:-2])
        
        self.att_16 = ARM(256)
        self.att_32 = ARM(512)
        self.global_ave_pooling = nn.AdaptiveAvgPool2d(1)

        # self.conv32 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        '''
        '''

        # tic = time.time()
        outs = []
        for _, m in enumerate(self.basenet):
            x = m(x)
            outs += [x]
        # print('resnet18: ', time.time() - tic)

        fea_att_16 = self.att_16(outs[-2])
        fea_att_32 = self.att_32(outs[-1])
        fea_global = self.global_ave_pooling(outs[-1])

        fea_att_32 = fea_att_32 + fea_global # * or +
        fea_att_32 = F.interpolate(fea_att_32, scale_factor=4, mode='bilinear')
        fea_att_16 = F.interpolate(fea_att_16, scale_factor=2, mode='bilinear')
        out = torch.cat([fea_att_16, fea_att_32, fea_global.expand_as(fea_att_32)], dim=1) # good options, fast convergence
        # out = torch.cat([outs[-3], fea_att_16, fea_att_32], dim=1) # inference fast, fea_global usage is different from above

        return out, fea_att_32, fea_att_16


class BiSeNet(nn.Module):
    def __init__(self, num_classes):
        super(BiSeNet, self).__init__()

        self.spatial = SpatialNet()
        self.context = ContextNet()
        self.ffm = FFM(256, 256+512+512)
        self.conv = nn.Conv2d(256 + 256+512+512, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        '''
        '''
        # tic = time.time()
        out_spatial = self.spatial(x)
        # print('spatial: ', time.time() - tic)
        
        # tic = time.time()
        out_context, _, _ = self.context(x) # _ used for auxiliary loss
        # print('context: ', time.time() - tic)
        
        # tic = time.time()
        feat = self.ffm(out_spatial, out_context)
        # print('ffm: ', time.time() - tic)

        feat = F.interpolate(feat, scale_factor=8, mode='bilinear') # very slow when using cpu.
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

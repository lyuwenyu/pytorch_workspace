import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.nn.init as init

from .config import cfg
from .ops_priors_bbox import PriorBox
from .build_ssd_layer import SSDLayerLoss 

def vgg(cfg, i=3, batch_norm=False):
    '''
    '''
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(in_channels, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]

    vgg_base = models.vgg16(pretrained=True).features
    for m, _m in zip(layers[:-5], vgg_base):
        if isinstance(m, nn.Conv2d):
            m.weight.data.copy_(_m.weight.data)
            m.bias.data.copy_(_m.bias.data)

        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.copy_(_m.weight.data)
            m.bias.data.copy_(_m.bias.data)

            # m.weight.requires_grad = False
            # m.bias.requires_grad = False

    return layers


def extra(cfg, i, batch_norm=False):
    '''
    '''
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k+1], kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag

        in_channels = v

    return layers


def multibox(vgg_layers, extra_layers, cfg, num_classes):
    '''
    '''
    loc_layers = []
    conf_layers = []
    # TODO HERE
    # if bn 
    # after relu [22, -1]
    # after conv [21, -2]
    vgg_source = [21, -2]
    
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg_layers[v].out_channels, cfg[k] * 4, kernel_size=3, stride=1, padding=1)]
        conf_layers += [nn.Conv2d(vgg_layers[v].out_channels, cfg[k] * num_classes, kernel_size=3, stride=1, padding=1)]

    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k] * 4, kernel_size=3, stride=1, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k] * num_classes, kernel_size=3, stride=1, padding=1)]
    
    return vgg_layers, extra_layers, (loc_layers, conf_layers)


class SSD(nn.Module):
    def __init__(self, base, extra, head, num_classes):
        super(SSD, self).__init__()
        self.num_classes = num_classes
        self.base = nn.ModuleList(base)
        self.extra = nn.ModuleList(extra)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.l2norm = L2Norm(512, 20)

        self.proirs = PriorBox()(VISUALIZATION=False)
        self.criteria = SSDLayerLoss()

    def forward(self, x, target=None):
        '''
        '''
        features = []
        loc = []
        conf = []
        priors = self.proirs.to(device=x.device)

        # conv4_3 relu
        num_conv43_relu = 22
        for k in range(num_conv43_relu + 1):
            x = self.base[k](x)

        _x = self.l2norm(x)
        features += [_x]
        # features += [x]
        
        for k in range(num_conv43_relu + 1, len(self.base)):
            x = self.base[k](x)

        features += [x]

        for k, v in enumerate(self.extra):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                features += [x]

        for x, l, c in zip(features, self.loc, self.conf):    
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        
        loc = torch.cat([o.view(o.size(0), -1, 4) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1, self.num_classes) for o in conf], 1)

        # print(loc.shape, conf.shape)
        
        if target is None:
            
            conf = F.softmax(conf, dim=-1)
            
            loc[:, :, :2] = (loc[:, :, :2] * cfg['variances'][0] * priors[:, 2:] + priors[:, :2])
            loc[:, :, 2:] = torch.exp(loc[:, :, 2:] * cfg['variances'][1]) * priors[:, 2:] # + priors[:, 2:]

            loc[:, :, :2] -= loc[:, :, 2:] / 2
            loc[:, :, 2:] += loc[:, :, :2]

            return conf, loc

        else:

            loss = self.criteria(loc, conf, priors, target)

            return loss


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale, format='spatial'):
        super(L2Norm, self).__init__()
        assert format in ('spatial', 'cross_channel')
        self.n_channels = n_channels
        self.gamma = scale
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self, ):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        '''
        shape: n, c, h, w
        '''
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = x / norm
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        
        return out


def build_ssd(size=300, num_classes=21):
    '''
    '''
    _base, _extra, _head = multibox(vgg(cfg['vgg'], i=3, batch_norm=False), extra(cfg['extra'], i=1024, batch_norm=False), cfg['mbox'], num_classes)
    return SSD(_base, _extra, _head, num_classes)


if __name__ == '__main__':

    dummy = torch.rand(3, 3, 300, 300)

    ssd = build_ssd()
    print(ssd)
    print(ssd(dummy))
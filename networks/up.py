import torch
import torch.nn as nn


class UP2x(nn.Module):
    def __init__(self, inc):
        self.m = nn.ConvTanspose2d(inc, inc, kernel_size=4, stride=2, padding=1, groups=inc)

    def forward(self, x):
        '''
        '''
        return self.m(x)



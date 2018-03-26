

import torch
import torch.nn as nn
from torch.autograd import Variable



class group_normalization(nn.Module):

    def __init__(self, channels_num, groups_num, eps=1e-6):
        super(group_normalization, self).__init__()

        self.groups_num = groups_num
        self.channels_num = channels_num

        self.gamma = nn.Parameter(torch.ones(1, channels_num, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels_num, 1, 1))

        self.eps = eps

    def forward(self, x):

        n,c,h,w = x.size()

        assert c == self.channels_num
        assert c % self.groups_num == 0


        x = x.view(n, self.groups_num, -1)

        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        # var = x.var(dim=-1, keepdim=True)

        x = (x-mean) / (std+self.eps)
        # x = (x-mean) / torch.sqrt(var+self.eps)

        x = x.view(n,c,h,w)

        x = x * self.gamma + self.beta

        return x





if __name__ == "__main__":

    mm = group_normalization(32, 8)

    x = Variable( torch.randn(10, 32, 10, 10))


   
    print(mm(x).size())

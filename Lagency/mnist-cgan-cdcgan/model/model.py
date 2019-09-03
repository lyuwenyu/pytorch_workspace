import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable
import copy


def _weight_init(mm):

    if isinstance(mm, nn.Linear):

        init.xavier_normal_(mm.weight.data)  
        mm.bias.data.zero_()


class generator(nn.Module):

    def __init__(self, z_size=100, y_size=10, num_layers=2):

        super(generator, self).__init__()

        models = []

        layer1 = nn.Linear(z_size+y_size, 128)
        models += [layer1, nn.ReLU()]

        layer1_1 = nn.Linear(128, 512)
        models += [layer1_1, nn.ReLU()]

        layer2 = nn.Linear(512, 784)
        models += [layer2, nn.Tanh()]

        self.models = nn.Sequential(*models)

        self.apply(_weight_init)


    def forward(self, x, y):

        out = torch.cat([x, y], dim=-1)

        out = self.models(out)

        return out




class discriminator(nn.Module):

    def __init__(self, x_size=784, y_size=10):

        super(discriminator, self).__init__()
        
        self.attention = Attention(512)

        models = []

        self.layer1 = nn.Linear(x_size+y_size, 512)
        # models += [layer1, nn.LeakyReLU(), attention]

        self.layer2 = nn.Linear(512, 128)
        # models += [layer2, nn.LeakyReLU()]

        self.layer3 = nn.Linear(128, 1)
        # models += [layer3, nn.Sigmoid()]

        # self.models = nn.ModuleList(models)

        self.apply(_weight_init)


    def forward(self, x, y, mem):

        out = torch.cat([x, y], dim=-1)

        out = F.leaky_relu(self.layer1(out))
        out, mem_tmp = self.attention(out, y, mem)
        out = F.leaky_relu(self.layer2(out))
        out = self.layer3(out)
        
        return F.sigmoid(out), mem_tmp


class Attention(nn.Module):
    def __init__(self, c=256, shortcut=True):
        super(Attention, self).__init__()
        self.shortcut = shortcut
        
        self.fc1 = nn.Linear(c, 128)
        self.fc2 = nn.Linear(128, c)
        
        self.alpha = 0.6

    def forward(self, x, y, mem):

        mem_tmp = torch.zeros_like(mem)

        q = self.fc1(x)
        p = F.softmax(torch.mm(q, mem.t()), dim=-1)
        att = (p[:, :, None] * mem[None, :, :]).sum(dim=1)

        for i in range(10):
            index = (torch.argmax(y, dim=1)==i).to(x.device, dtype=x.dtype)
            mem_tmp[i] = self.alpha * mem[i] + (1-self.alpha) * torch.mean(index[:, None] * att, dim=0)

        out = self.fc2(att)
        
        if self.shortcut:
            out += x

        return out, mem_tmp
        


if __name__ == '__main__':
    

    gg = generator()
    dd = discriminator()

    x = Variable( torch.rand(3, 784) )
    z = Variable( torch.rand(3, 100) )
    y = Variable( torch.rand(3, 10 ) )
    print( gg(z, y).size())
    print( dd(x, y).size())
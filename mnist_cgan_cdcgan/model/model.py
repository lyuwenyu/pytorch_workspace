import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable



def _weight_init(mm):

    if isinstance(mm, nn.Linear):

        init.xavier_normal(mm.weight.data)  
        mm.bias.data.zero_()


class generator(nn.Module):

    def __init__(self, z_size=100, y_size=10, num_layers=2):

        super(generator, self).__init__()

        
        self.layer1 = nn.Linear(z_size+y_size, 128)
        self.layer2 = nn.Linear(128, 784)


        self.apply(_weight_init)


    def forward(self, x, y):

        out = torch.cat([x, y], dim=-1)

        out = self.layer1(out)
        out = F.relu(out)

        out = self.layer2(out)
        out = F.tanh(out)

        return out






class discriminator(nn.Module):

    def __init__(self, x_size=784, y_size=10):

        super(discriminator, self).__init__()

        self.layer1 = nn.Linear(x_size+y_size, 128)
        self.layer2 = nn.Linear(128, 1)


        self.apply(_weight_init)


    def forward(self, x, y):

        out = torch.cat([x, y], dim=-1)

        out = self.layer1(out)
        out = F.leaky_relu(out)

        out = self.layer2(out)
        out = F.sigmoid(out)

        return out



if __name__ == '__main__':
    

    gg = generator()
    dd = discriminator()


    x = Variable( torch.rand(3, 784) )
    z = Variable( torch.rand(3, 100) )
    y = Variable( torch.rand(3, 10 ) )
    print( gg(z, y).size())
    print( dd(x, y).size())
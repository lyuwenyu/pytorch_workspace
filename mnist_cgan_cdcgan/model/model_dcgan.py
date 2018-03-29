import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.autograd import Variable

def _weight_init(mm):
    pass


class generator(nn.Module):
    
    def __init__(self, z_size=100, y_size=10):
        super(generator, self).__init__()

        self.deconv1 = nn.ConvTranspose2d(z_size+y_size, 256, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(256)

        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(128)

        self.deconv3 = nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2)



    def forward(self, z, y):

        out = torch.cat([z, y], dim=1)  # [batch, 110, 1, 1]

        out = F.relu( self.bn1( self.deconv1(out) ) )
        # out = self.bn1(out)

        out = F.relu( self.bn2( self.deconv2(out) ) )
        # out = self.bn2(out)

        out = self.deconv3(out)

        return F.tanh(out)



class discriminator(nn.Module):

    def __init__(self, x_size=1, y_size=10):
        super(discriminator, self).__init__()

        self.conv1 = nn.Conv2d(11, 128, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(128, 256, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(256, 1, kernel_size=4, stride=1)


    def forward(self, x, y): # n 1 28 28,  n 10 28 28

        out = torch.cat([x, y], dim=1)

        out = F.relu( self.bn1( self.conv1(out) ) )

        out = F.relu( self.bn2( self.conv2(out) ) )

        out = self.conv3(out)

        return F.sigmoid(out)



if __name__ == '__main__':


    gg = generator()
    dd = discriminator()

    x = Variable( torch.randn(10, 1, 28, 28))
    z = Variable( torch.randn(10, 100, 1, 1) )
    y = Variable( torch.randn(10, 10, 1, 1) )
    y_ = y + Variable( torch.zeros(10, 10, 28,28) )

    print(gg(z, y).size())
    print(dd(x, y_).size())
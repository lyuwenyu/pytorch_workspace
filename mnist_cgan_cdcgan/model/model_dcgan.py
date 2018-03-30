import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.autograd import Variable
import torch.nn.init as init 
from .gn import group_normalization 


def _weight_init(mm):
    
    if isinstance(mm, nn.Linear):
        init.xavier_normal(mm.weight.data)
        mm.bias.data.zero_()

    elif isinstance(mm, nn.Conv2d):
        init.xavier_normal(mm.weight.data)
        mm.bias.data.zero_()

    elif isinstance(mm, nn.ConvTranspose2d):
        init.xavier_normal(mm.weight.data)
        mm.bias.data.zero_()

    elif isinstance(mm, group_normalization):
        pass


class generator(nn.Module):
    
    def __init__(self, z_size=100, y_size=10):
        super(generator, self).__init__()


        self.deconv1 = nn.ConvTranspose2d(z_size+y_size, 256, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(256)
        # self.gn1 = group_normalization(256, 8)

        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        # self.gn2 = group_normalization(128, 4)

        self.deconv3 = nn.ConvTranspose2d(128, 1, kernel_size=3, stride=2)


        ### generator version_2, project and reshape firstly

        # self.fc1 = nn.Linear(z_size+y_size, 3*3*512)
        # self.new_deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=7, stride=2)
        # self.new_bn1 = nn.BatchNorm2d(256)
        # self.new_deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2)
        # self.new_bn2 = nn.BatchNorm2d(128)
        # self.new_deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1)
        # self.new_bn3 = nn.BatchNorm2d(64)
        # self.new_deconv4 = nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1)


        self.apply(_weight_init)




    def forward(self, z, y):

        
        out = torch.cat([z, y], dim=1)  # [batch, 110, 1, 1]
        out = F.leaky_relu( self.bn1( self.deconv1(out) ), negative_slope=0.2 )
        out = F.leaky_relu( self.bn2( self.deconv2(out) ), negative_slope=0.2 )
        out = self.deconv3(out)
        out = F.pad(out, pad=[1,0,1,0], mode='reflect')  ##  make sure h w equel to 28
        return F.tanh(out)
        

        ### generator version_2, project and reshape firstly

        # out = torch.cat([z, y], dim=1)
        # out = torch.squeeze(out)
        # out = self.fc1(out)
        # out = out.view(out.size()[0], 512, 3, 3)
        # out = F.leaky_relu( self.new_bn1( self.new_deconv1(out) ) )
        # out = F.leaky_relu( self.new_bn2( self.new_deconv2(out) ) )
        # out = F.leaky_relu( self.new_bn3( self.new_deconv3(out) ) )
        # out = self.new_deconv4(out)
        # out = F.tanh(out)
        # return out[:,:,1:,1:]




class discriminator(nn.Module):

    def __init__(self, x_size=1, y_size=10):
        super(discriminator, self).__init__()

        self.conv1 = nn.Conv2d(11, 64, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.gn1 = group_normalization(64, 2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.gn2 = group_normalization(128, 4)

        self.fc = nn.Linear(2048, 1)

        self.apply(_weight_init)

    def forward(self, x, y): # n 1 28 28,  n 10 28 28

        out = torch.cat([x, y], dim=1)

        out = F.leaky_relu( self.bn1( self.conv1(out) ), negative_slope=0.2 )
        out = F.leaky_relu( self.bn2( self.conv2(out) ), negative_slope=0.2 )

        out = self.fc(out.view(-1, 2048))


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
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

### unet mdoel

class UnetSkipConnectionBlock(nn.Module):

    def __init__(self, inner_nc, outer_nc, input_nc=None, submodule=None, innermot=False, outermost=False, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()

        pass

    
    def forward(self, x):
        pass




if __name__ == '__main__':

    pass





### pixel discriminator


class n_layer_discriminator(nn.Module):

    def __init__(self, ):
        super(n_layer_discriminator, self).__init__()


        pass


    def forward(self, x):
        pass


        
class pixel_discriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, use_sigmoid=False, gpu_ids=[]):
        super(pixel_discriminator, self).__init__()

        self.gpu_ids = gpu_ids

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf*2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, 1, kernel_size=1, stride=1, padding=0)
        ]

        if use_sigmoid:

            self.net.append( nn.Sigmoid() )

        self.net = nn.Sequential( *self.net )


    def forward(self, x):

        if len(self.gpu_ids) and isinstance(x.data, torch.cuda.FloatTensor):

            return nn.parallel.data_parallel(self.net, x, device_ids=self.gpu_ids)

        return self.net(x)


if __name__ == '__main__':

    m1 = pixel_discriminator(3, use_sigmoid=True)

    x1 = Variable( torch.randn(10, 3, 64, 64) )

    print(m1)
    print(m1(x1).size())



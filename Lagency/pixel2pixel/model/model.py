import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


### loss

class ganloss(nn.Module):
    
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor):

        super(ganloss, self).__init__()

        self.real_label = target_real_label
        self.fake_label = target_fake_label

        self.real_label_var = None
        self.fake_label_var = None

        self.tensor = tensor

        if use_lsgan:

            self.loss = nn.MSELoss()

        else:

            self.loss = nn.BCELoss()


    def get_target_tensor(self, input, target_is_real):

        target_tensor = None

        if target_is_real:

            create_label = ((self.real_label_var is None) or (self.real_label_var.numel() != input.numel()))

            if create_label:

                real_tensor = self.tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)

            target_tensor = self.real_label_var

        else:

            create_label = ((self.fake_label_var is None) or (self.fake_label_var.numel() != input.numel()))

            if create_label:

                real_tensor = self.tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(real_tensor, requires_grad=False)
                
            target_tensor = self.fake_label_var


        return target_tensor

    
    def __call__(self, input, target_is_real):

        target_tensor = self.get_target_tensor(input, target_is_real)

        return self.loss(input, target_tensor)





### unet mdoel

class UnetSkipConnectionBlock(nn.Module):

    def __init__(self, inner_nc, outer_nc, input_nc=None, submodule=None, innermost=False, outermost=False, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()

        self.outermost = outermost

        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1)
        downrelu = nn.LeakyReLU(0.2)
        downnorm = nn.BatchNorm2d(inner_nc)

        uprelu = nn.ReLU()
        upnorm = nn.BatchNorm2d(outer_nc)

        if outermost:

            upconv = nn.ConvTranspose2d(inner_nc*2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downrelu, downconv]
            up = [uprelu, upconv, nn.Tanh()]

            model = down + [submodule] + up


        elif innermost:

            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1)

            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]

            model = down + up

        else:

            upconv = nn.ConvTranspose2d(inner_nc*2, outer_nc, kernel_size=4, stride=2, padding=1)

            down = [downrelu, downconv, downnorm]
            up = [downrelu, downconv, downnorm]

            if use_dropout:
                
                model = down + [submodule] + up + [nn.Dropout(0.5)]

            else:

                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        
        if self.outermost:

            return self.model(x)

        else:

            return torch.cat([x, self.model(x)], dim=1)


class unet(nn.Module):

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, use_dropout=False):
        super(unet, self).__init__()

        unet_block = UnetSkipConnectionBlock(ngf*8, ngf*8, innermost=True)

        for i in range(num_downs-5):
            unet_block = UnetSkipConnectionBlock(ngf*8, ngf*8, submodule=unet_block)

        unet_block = UnetSkipConnectionBlock(ngf*4, ngf*8, submodule=unet_block)
        unet_block = UnetSkipConnectionBlock(ngf*2, ngf*4, submodule=unet_block)
        unet_block = UnetSkipConnectionBlock(ngf, ngf*2, submodule=unet_block)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True)

        self.unet = unet_block

    
    def forward(self, x):
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

    print('-----')


    crit = ganloss()
    x1 = Variable(torch.randn(10, 1, 224,224))
    loss = crit(x1, target_is_real=False)
    print(loss)


    print('----')


    mm = unet(3, 1, 7)
    print(mm)
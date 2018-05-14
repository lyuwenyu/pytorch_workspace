import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from model import model_dcgan 
import torch.utils.data as data 
import numpy as np
from data.dataset import MNIST
import torchvision.utils as vutils

batch_size = 50

mnist = MNIST()
dataloader = data.DataLoader(mnist, batch_size=batch_size, shuffle=True)

gnet = model_dcgan.generator()
dnet = model_dcgan.discriminator()

if torch.cuda.is_available():
    gnet = gnet.cuda()
    dnet = dnet.cuda()

g_optim = optim.Adam(gnet.parameters(), lr=0.0002, betas=(0.5, 0.99))
d_optim = optim.Adam(dnet.parameters(), lr=0.0002, betas=(0.5, 0.99))

crit = nn.BCELoss()
if torch.cuda.is_available():
    crit = crit.cuda()

fixed_z = Variable( torch.randn(100, 100, 1, 1))
fixed_y = Variable( torch.cat([torch.eye(10,10)]*10, dim=0).view(100, 10, 1, 1))
# fixed_y_exp = Variable( fixed_y + Variable(torch.zeros(100, 10, 28, 28)))
if torch.cuda.is_available():
    fixed_y = fixed_y.cuda()
    fixed_z = fixed_z.cuda()

l1loss = nn.L1Loss()

for e in range(100):

    # dnet.train()
    

    for i, (imgs, labs) in enumerate(dataloader):

        gnet.train()

        # train d
        for _ in range(2):

            x = Variable(imgs.float().view(batch_size, 1, 28,28))


            y = Variable(labs.float().view(batch_size, 10, 1, 1))
            y_exp = Variable( torch.zeros(batch_size, 10, 28,28) ) + y
            # z = Variable( torch.randn(batch_size, 100, 1, 1))
            z = Variable( (torch.rand(batch_size, 100, 1, 1)-0.5)/0.5 )

            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
                y_exp = y_exp.cuda()
                z = z.cuda()


            d_real = dnet(x, y_exp)
            
            fake = gnet(z, y)
            # fake.detach()
            d_fake = dnet(fake.detach(), y_exp)

            if torch.cuda.is_available():
                d_loss = crit(d_real, Variable( torch.ones(batch_size,1).float().cuda() ) )  + \
                            crit(d_fake, Variable( torch.zeros(batch_size,1).float().cuda() ))
            else:
                d_loss = crit(d_real, Variable( torch.ones(batch_size,1).float() ) )  + \
                            crit(d_fake, Variable( torch.zeros(batch_size,1).float() ))

            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()


        # train g

        # # z = Variable( torch.randn(batch_size, 100, 1, 1))
        # z = Variable( (torch.rand(batch_size, 100, 1, 1)-0.5)/0.5 )

        # y = torch.eye(10)
        # index = np.random.randint(0,10, (batch_size))
        # y = y[ torch.from_numpy(index) ]
        # y = Variable( y.view(batch_size, 10, 1, 1) )

        # y_exp = Variable( torch.zeros(batch_size, 10, 28,28) ) + y


        # if torch.cuda.is_available():
        #     z = z.cuda()
        #     y = y.cuda()
        #     y_exp = y_exp.cuda()

        # fake = gnet(z, y)
        d_fake = dnet(fake, y_exp)

        if torch.cuda.is_available():

            g_loss = crit(d_fake, Variable( torch.ones(batch_size,1).float().cuda() )) + l1loss(fake, x)

        else:

            g_loss = crit(d_fake, Variable( torch.ones(batch_size,1).float() ))

        g_optim.zero_grad()
        g_loss.backward()
        g_optim.step()


        if i%20 == 0:
            print(e, i, d_loss.cpu().data.numpy(), g_loss.cpu().data.numpy())

            gnet.eval()
            fake = gnet(fixed_z, fixed_y)
            vutils.save_image( fake.cpu().data , './output/gn_dcgan_{:0>3}.jpg'.format(e), nrow=10)
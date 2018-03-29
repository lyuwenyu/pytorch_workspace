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

g_optim = optim.Adam(gnet.parameters(), lr=0.002)
d_optim = optim.Adam(dnet.parameters(), lr=0.002)

crit = nn.BCELoss()


fixed_z = Variable( torch.randn(100, 100, 1, 1))
fixed_y = Variable( torch.cat([torch.eye(10,10)]*10, dim=0).view(100, 10, 1, 1))
# fixed_y_exp = Variable( fixed_y + Variable(torch.zeros(100, 10, 28, 28)))

for e in range(100):

    # dnet.train()
    

    for i, (imgs, labs) in enumerate(dataloader):

        gnet.train()

        # train d
        for _ in range(2):

            x = Variable(imgs.float().view(batch_size, 1, 28,28))
            y = Variable(labs.float().view(batch_size, -1, 1, 1))
        
            y_exp = Variable( torch.zeros(batch_size, 10, 28,28) ) + y
            z = Variable( torch.randn(batch_size, 100, 1, 1))

            d_real = dnet(x, y_exp)
            
            fake = gnet(z, y)
            fake.detach()
            d_fake = dnet(fake, y_exp)

            d_loss = crit(d_real, Variable( torch.ones(batch_size,1,1,1).float() ) )  + \
                        crit(d_fake, Variable( torch.zeros(batch_size,1,1,1).float() ))

            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()


        # train g

        z = Variable( torch.randn(batch_size, 100, 1, 1))

        y = torch.eye(10)
        index = np.random.randint(0,10, (batch_size))
        y = y[ torch.from_numpy(index) ]
        y = Variable( y.view(batch_size, -1, 1, 1) )

        y_exp = Variable( torch.zeros(batch_size, 10, 28,28) ) + y


        fake = gnet(z, y)
        d_fake = dnet(fake, y_exp)

        g_loss = crit(d_fake, Variable( torch.ones(batch_size,1,1,1).float() ))


        g_optim.zero_grad()
        g_loss.backward()
        g_optim.step()


        if i%50 == 0:
            print(e, i, d_loss.data.numpy(), g_loss.data.numpy())

            gnet.eval()
            fake = gnet(fixed_z, fixed_y)
            vutils.save_image( fake.data , './output/dcgan_{:0>3}.jpg'.format(e), nrow=10)
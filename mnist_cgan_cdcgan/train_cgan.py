
from model import model
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from PIL import Image
import torchvision.utils as vutils
import numpy as np 
from data.dataset import MNIST
import torch.utils.data as data 



fixed_z = Variable( torch.randn(100, 100) )
fixed_y = Variable( torch.cat([torch.eye(10, 10)]*10, dim=0) )

batch_size = 100

mnist = MNIST()
dataloader = data.DataLoader(mnist, batch_size=batch_size, shuffle=True)


print(fixed_y.shape, fixed_z.shape)


gnet = model.generator(z_size=100, y_size=10)
dnet = model.discriminator(x_size=784, y_size=10)

g_optim = optim.Adam(gnet.parameters(), lr=0.002)
d_optim = optim.Adam(dnet.parameters(), lr=0.002)

crit = nn.BCELoss()

def train():


    for e in range(100):

        for i, (imgs, labs)  in  enumerate(dataloader):


            ### train d

            x = Variable( imgs.float() )
            y = Variable( labs.float() )
            z = Variable( torch.randn(batch_size, 100))

            d_real = dnet(x, y)

            fake = gnet(z, y)
            d_fake = dnet(fake, y)


            d_loss = crit(d_real, Variable( torch.ones(batch_size,1).float() ) ) \
                    + crit(d_fake, Variable( torch.zeros(batch_size,1).float() ))

            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()


            ### train g
            
            y = torch.eye(10)
            index = np.random.randint(0,10, (batch_size))
            y = y[ torch.from_numpy(index) ]
            y = Variable( y )
            # y = Variable( torch.from_numpy(y).float() )
            z = Variable( torch.randn(batch_size, 100))

            fake = gnet(z, y)
            d_fake = dnet(fake, y)

            g_loss = crit(d_fake, Variable( torch.ones(batch_size,1).float()) )

            g_optim.zero_grad()
            g_loss.backward()
            g_optim.step()


            if i % 10 == 0:
                print(e, i, d_loss.data.numpy(), g_loss.data.numpy())

                
        fake = gnet(fixed_z, fixed_y)
        vutils.save_image(fake.data.view(100, 1, 28, 28), nrow=10, filename='./output/test_1_{:0>3}.jpg'.format(e))


if __name__ == '__main__':

    train()
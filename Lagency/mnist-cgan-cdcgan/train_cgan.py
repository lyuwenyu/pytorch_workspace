
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
import random

device = torch.device('cuda:1')

# fixed_z = torch.from_numpy(np.random.uniform(-1, 1, (100, 100))).to(device, dtype=torch.float32)
# fixed_z = torch.randn(100, 100).to(device)
fixed_z = (2*(torch.rand(100, 100)-0.5)).to(device)
fixed_y = torch.cat([torch.eye(10, 10)]*10, dim=0).to(device, dtype=torch.float32)

batch_size = 100

mnist = MNIST()
dataloader = data.DataLoader(mnist, batch_size=batch_size, shuffle=True)

n = 3
for i, (im, la)  in  enumerate(dataloader):
    fixed_img_0 = im[la == 0][:n]
    fixed_lab_0 = la[la == 0][:n]
    print(fixed_lab_0)
    break


gnet = model.generator(z_size=100, y_size=10).to(device)
dnet = model.discriminator(x_size=784, y_size=10).to(device)

g_optim = optim.Adam(gnet.parameters(), lr=0.002)
d_optim = optim.Adam(dnet.parameters(), lr=0.002)

crit = nn.BCELoss().to(device)


def train():

    mem = torch.zeros(10, 128, requires_grad=False).to(device)

    for n in range(5, 10):

        for e in range(100):

            for i, (imgs, labs)  in  enumerate(dataloader):

                inx = labs > 0
                imgs = imgs[inx]
                labs = labs[inx]
                
                if random.random() < 0.6 and e > 5:
                    imgs = torch.cat([imgs, fixed_img_0], dim=0)
                    labs = torch.cat([labs, fixed_lab_0], dim=0)
                
                labs = torch.eye(10)[labs]
                b = imgs.shape[0]

                ### train d
                for _ in range(1):

                    x = imgs.view(-1, 28*28).to(device)
                    y = labs.to(device)
                    z = (2*(torch.rand(b, 100)-0.5)).to(device)

                    d_real, mem_tmp = dnet(x, y, mem)

                    fake = gnet(z, y)
                    d_fake, _ = dnet(fake.detach(), y, mem)

                    d_loss = crit(d_real, torch.ones(b,1).to(device, dtype=d_real.dtype)) \
                            + crit(d_fake, torch.zeros(b,1).to(device, dtype=d_fake.dtype) )

                    d_optim.zero_grad()
                    d_loss.backward()
                    d_optim.step()

                    mem.data.copy_(mem_tmp.data)

                ### train g
                y = (torch.eye(10)[torch.randint(0, 10, (b, )).long()]).to(device)
                z = (2*(torch.rand(b, 100)-0.5)).to(device)

                fake = gnet(z, y)
                d_fake, mem_tmp = dnet(fake, y, mem)

                g_loss = crit(d_fake, torch.ones(b, 1).to(device, dtype=d_fake.dtype))

                g_optim.zero_grad()
                g_loss.backward()
                g_optim.step()
                
                if e > n:
                    mem.data.copy_(mem_tmp.data)

                if i % 10 == 0:
                    print(e, i, d_loss.cpu().item(), g_loss.cpu().item())

                    
            fake = gnet(fixed_z, fixed_y)
            vutils.save_image(fake.cpu().data.view(100, 1, 28, 28), nrow=10, filename='./output/test_att_{}_{:0>3}.jpg'.format(n, e))


if __name__ == '__main__':

    train()
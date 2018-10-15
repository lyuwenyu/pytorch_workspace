import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data 
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from model.network import BiSeNet
from data.dataset import Dataset

device = torch.device('cuda:0')

import argparse
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
parser = argparse.ArgumentParser()
parser.add_argument('--device_ids', type=list, default=[0, 1])
parser.add_argument('--num_classes', type=int, default=20)
parser.add_argument('--resume_path', type=str, default='') # ./output/state-ckpt-epoch-00000
parser.add_argument('--loss_step', type=int, default=10)
parser.add_argument('--save_step', type=int, default=5)
parser.add_argument('--img_dims', type=list, default=[416, ])
parser.add_argument('--batch_sizes', type=list, default=[32, ])
parser.add_argument('--num_workers', type=list, default=8)
parser.add_argument('--epoches', type=int, default=201)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--momentum', type=float, default=0.95)
parser.add_argument('--milestones', type=list, default=[80, 150, 180])
parser.add_argument('--gamma', type=float, default=0.1)
args = parser.parse_args()


if __name__ == '__main__':

    model = BiSeNet(num_classes=2)
    dataset = Dataset('./data')
    dataloader = data.DataLoader(dataset, batch_size=2, num_workers=1)

    model.to(device=device)
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.95)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=0.1)

    for e in range(100):
        for _, (imgs, msks) in enumerate(dataloader):

            # print(imgs.shape, msks.shape)

            imgs = imgs.to(device=device)
            msks = msks.to(device=device, dtype=torch.long)

            logits = model(imgs)

            loss = F.cross_entropy(logits, msks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print('loss: ', loss.item())

    torch.save(model.state_dict(), './output/state-0000')
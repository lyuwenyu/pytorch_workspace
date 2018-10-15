import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data 
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from model.network import BiSeNet
from data.dataset import Dataset

device = torch.device('cuda:0')


if __name__ == '__main__':

    model = BiSeNet(num_classes=2)
    dataset = Dataset('./data')
    dataloader = data.DataLoader(dataset, batch_size=2, num_workers=1)

    model.to(device=device)
    
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.95)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=0.1)

    for e in range(10):
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
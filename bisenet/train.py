import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data 
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import numpy as np
import time
import os
import sys
sys.path.insert(0, '/home/wenyu/workspace/pytorch_workspace/')
import yolov3.utils.ops_weight_init as ops_weight_init
from model.network import BiSeNet
from data.dataset import Dataset


device = torch.device('cuda:0')

import argparse
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
parser = argparse.ArgumentParser()
parser.add_argument('--device_ids', type=list, default=[0, 1])
parser.add_argument('--num_classes', type=int, default=5)
parser.add_argument('--resume_path', type=str, default='')
parser.add_argument('--loss_step', type=int, default=10)
parser.add_argument('--save_step', type=int, default=50)
parser.add_argument('--img_dims', type=list, default=[416, ])
parser.add_argument('--batch_sizes', type=list, default=[32, ])
parser.add_argument('--num_workers', type=list, default=8)
parser.add_argument('--epoches', type=int, default=201)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--momentum', type=float, default=0.95)
parser.add_argument('--milestones', type=list, default=[80, 150, 180])
parser.add_argument('--gamma', type=float, default=0.1)
args = parser.parse_args()


def get_weight(msks):
    '''weighted for loss'''
    num = [torch.sum(msks == i).item() for i in range(args.num_classes)]
    _num = []
    for n in num:
        if n == 0:
            _num += [0]
        else:
            _num += [sum(num) / n]
    weight = torch.tensor([n/sum(_num) for n in _num]).to(device=device, dtype=torch.float32)

    return weight


def train(model, dataloader, optimizer, scheduler, epoch=0):
    '''train'''
    model.train()
    scheduler.step()

    step_losses = []
    step_time = []

    for i, (images, targets) in enumerate(dataloader):
        images = images.to(device=device)
        targets = targets.to(device=device, dtype=torch.long)

        weight = get_weight(targets)
        tic = time.time()
        logits = model(images)
        losses = F.cross_entropy(logits, targets, weight=weight)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        step_losses += [losses.item()]
        step_time += [time.time() - tic]

        if (i + 1) % args.loss_step == 0:

            step_losses = np.array(step_losses).mean(axis=0)

            lin = 'epoch/iter: {:0>3}/{:0>3}, lr: {}, time: {:.4f}s, losses: {:.4f}'.format(e, i + 1,
                                                                                format(optimizer.param_groups[0]['lr'],'.0e'),
                                                                                np.array(step_time).mean(),
                                                                                step_losses)
            step_losses, step_time = [], []
            print(lin)
            
    if e % args.save_step == 0:
        state = {
            'epoch': e,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        torch.save(state, './output/state-ckpt-epoch-{:0>5}'.format(epoch))


if __name__ == '__main__':

    model = BiSeNet(num_classes=args.num_classes)
    dataset = Dataset('./data')
    dataloader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model.to(device=device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    if args.resume_path:
        ops_weight_init.resume(model, optimizer, scheduler, args.resume_path)

    for e in range(scheduler.last_epoch, args.epoches):
        for _, (imgs, msks) in enumerate(dataloader):

            train(model, dataloader, optimizer, scheduler)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import time
import random
import glob
import numpy as np

from data.dataset import Dataset
from model.build_model import DarkNet
from utils import ops_weight_init

import logging
logger = logging.getLogger('train')

import argparse
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--device_ids', type=list, default=[0, 1])
parser.add_argument('--num_classes', type=int, default=20)
parser.add_argument('--resume_path', type=str, default='') # ./output/state-ckpt-epoch-00000
parser.add_argument('--loss_step', type=int, default=10)
parser.add_argument('--save_step', type=int, default=5)
parser.add_argument('--img_dims', type=list, default=[416, ])
parser.add_argument('--batch_sizes', type=list, default=[16, ])
parser.add_argument('--num_workers', type=list, default=8)
parser.add_argument('--epoches', type=int, default=201)
parser.add_argument('--model_cfg', type=str, default='./_model/yolov3.cfg')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--milestones', type=list, default=[80, 150, 180])
parser.add_argument('--gamma', type=float, default=0.1)
args = parser.parse_args()

def train(model, dataloader, optimizer, scheduler, epoch=0):
    '''train'''
    model.train()
    scheduler.step()

    step_losses = []
    step_time = []

    for i, (images, targets) in enumerate(dataloader):
        images = images.to(torch.device(device))
        targets = targets.to(torch.device(device))

        tic = time.time()
        losses = model(images, targets)
        # losses = nn.parallel.data_parallel(module=model, inputs=(images, targets), device_ids=args.device_ids, dim=0)
        losses = losses.sum(dim=0)
        step_losses += [losses.view(1, -1)]

        optimizer.zero_grad()
        losses[0].backward()
        optimizer.step()

        step_time += [time.time() - tic]

        if (i + 1) % args.loss_step == 0:
            # vs = np.array([v for losses in step_losses for v in losses.values()]).reshape(args.loss_step, -1).mean(axis=0)
            # lin = ', '.join(['{}: {:0.6f}'.format(k, v) for k, v in zip(losses.keys(), vs)])
            step_losses = torch.cat(step_losses).mean(dim=0)
            lin = ', '.join(['{:0.6f}'.format(ls.item()) for ls in step_losses])
            lin = 'epoch/iter: {:0>3}/{:0>3}, lr: {}, time(s): {:.4f}s, losses: {}'.format(e, i + 1,
                                                                                format(optimizer.param_groups[0]['lr'],'.0e'),
                                                                                np.array(step_time).mean(),
                                                                                lin)
            step_losses, step_time = [], []
            print(lin)


    if e % args.save_step == 0:
        state = {
            'epoch': e,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        # torch.save(model.state_dict(), './output/ckpt-epoch-{:0>5}'.format(epoch))
        torch.save(state, './output/state-ckpt-epoch-{:0>5}'.format(epoch))

    print(''.join(['-'] * len(lin)))


def validate(model, dataloader, epoch=0):
    model.eval()
    pass



if __name__ == '__main__':

    # dataset = Dataset('', size=args.img_dim)
    # dataloader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    # dataloaders = [dataloader]
    datasets = [Dataset('', size=sz) for sz in args.img_dims]
    dataloaders = [data.DataLoader(d, batch_size=bs, shuffle=True, num_workers=args.num_workers) for d, bs in zip(datasets, args.batch_sizes)]

    model = DarkNet(args.model_cfg, cls_num=args.num_classes)
    # model.load_weights('./_model/yolov3.weights')
    model = model.to(torch.device(device))

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    if args.resume_path:
        ops_weight_init.resume(model, optimizer, scheduler, args.resume_path)

    for e in range(scheduler.last_epoch + 1, int(args.epoches / len(dataloaders) + 1)):

        random.shuffle(dataloaders)
        
        for i, dtloader in enumerate(dataloaders):
            train(model, dtloader, optimizer, scheduler, e * len(dataloaders) + i)
        
        # train(model, dataloader, optimizer, scheduler, epoch=e)
        
        # validate(model, dataloader, epoch=e)


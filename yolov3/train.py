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
from utils import ops_logger

import argparse
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
parser = argparse.ArgumentParser()
parser.add_argument('--device_ids', type=list, default=[0, 1])
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--resume_path', type=str, default='') # ./output/state-ckpt-epoch-00000
parser.add_argument('--loss_step', type=int, default=10)
parser.add_argument('--save_step', type=int, default=10)
parser.add_argument('--img_dims', type=list, default=[416, 320, 512]) # 608, 320
parser.add_argument('--batch_sizes', type=list, default=[16, 24, 12]) # 16, 48
parser.add_argument('--num_workers', type=list, default=8)
parser.add_argument('--epoches', type=int, default=201)
parser.add_argument('--model_cfg', type=str, default='./_model/yolov3_v1.cfg')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--milestones', type=list, default=[100, 150, 180])
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
        
        # if 'cuda' in device and len(args.device_ids) > 1:
        #     losses = nn.parallel.data_parallel(module=model, inputs=(images, targets), device_ids=args.device_ids, dim=0)
        #     losses = losses.sum()
        # else:
        #     losses = model(images, targets)
        losses = model(images, targets)
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        step_losses += [losses.item()]
        step_time += [time.time() - tic]

        ops_logger.writer.add_scalar('losses/loss', losses.item(), global_step=e * len(dataloader) + i)

        if (i + 1) % args.loss_step == 0:

            step_losses = np.array(step_losses).mean(axis=0)
            # lin = ', '.join(['{:0.6f}'.format(ls) for ls in step_losses])
            lin = 'epoch/iter: {:0>3}/{:0>3}, lr: {}, time: {:.4f}s, losses: {:.4f}'.format(epoch, i + 1,
                                                                                format(optimizer.param_groups[0]['lr'],'.0e'),
                                                                                np.array(step_time).mean(),
                                                                                step_losses)
            step_losses, step_time = [], []
            print(lin)
            
    if epoch % args.save_step == 0:
        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        torch.save(state, './output/state-ckpt-epoch-{:0>5}'.format(epoch))

    print(''.join(['-'] * len(lin)))


def validate(model, dataloader, epoch=0):
    model.eval()
    pass



if __name__ == '__main__':

    # dataset = Dataset('', size=args.img_dim)
    # dataloader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    datasets = [Dataset('', size=sz) for sz in args.img_dims]
    dataloaders = [data.DataLoader(d, batch_size=bs, shuffle=True, num_workers=args.num_workers) for d, bs in zip(datasets, args.batch_sizes)]

    model = DarkNet(args.model_cfg, cls_num=args.num_classes)
    model.load_weights('./_model/yolov3.weights')
    model = model.to(torch.device(device))

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    if args.resume_path:
        ops_weight_init.resume(model, optimizer, scheduler, args.resume_path)

    # for e in range(scheduler.last_epoch + 1, int(args.epoches / len(dataloaders) + 1)):
    #     if e > 10: 
    #         random.shuffle(dataloaders)
    #     for i, dtloader in enumerate(dataloaders):
    #         train(model, dtloader, optimizer, scheduler, e * len(dataloaders) + i)

    e = scheduler.last_epoch + 1

    while e < args.epoches:
        if e < 10: 
            train(model, dataloaders[0], optimizer, scheduler, e)
            e += 1
        else:
            random.shuffle(dataloaders)
            for i, dtloader in enumerate(dataloaders):
                train(model, dtloader, optimizer, scheduler, e + i)
            e += len(dataloaders)

        # train(model, dataloader, optimizer, scheduler, epoch=e)
        
        # validate(model, dataloader, epoch=e)


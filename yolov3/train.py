import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import random
import glob
import numpy as np

from data.dataset import Dataset
from model.build_model import DarkNet
from utils.ops_weight_init import weight_init

import logging
logger = logging.getLogger('train')

import argparse
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--loss_step', type=int, default=10)
parser.add_argument('--save_step', type=int, default=10)
parser.add_argument('--img_dim', type=int, default=320)
parser.add_argument('--num_classes', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=3)
parser.add_argument('--epoches', type=int, default=121)
parser.add_argument('--model_cfg', type=str, default='./_model/yolov3.cfg')
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--momentum', type=float, default=0.99)
parser.add_argument('--milestones', type=list, default=[50, 80, 100])
args = parser.parse_args()

def train(model, dataloader, optimizer, epoch=0):

    model.train()
    step_losses = []

    for i, (images, target) in enumerate(dataloader):
        images = images.to(torch.device(device))
        target = target.to(torch.device(device))

        def get_target():
            bboxes = []
            for ii in range(target.shape[0]):
                _target = target[ii]
                _target = _target[_target[:, 0] > 0]
                bboxes += [_target[:, 1:]]
            return bboxes 

        target = get_target()
        loss, losses = model(images, target)
        
        step_losses += [losses]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % args.loss_step == 0:
            vs = np.array([v for losses in step_losses for v in losses.values()]).reshape(args.loss_step, -1).mean(axis=0)
            lin = ', '.join(['{}: {:0.6f}'.format(k, v) for k, v in zip(losses.keys(), vs)])
            lin = 'epoch/iter: {:0>3}/{:0>3}, lr: {}, {}'.format(e, i + 1, format(optimizer.param_groups[0]['lr'], '.0e'), lin)
            print(lin)

    if e % args.save_step == 0:
        state = {
            'epoch': e,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(model.state_dict(), './output/ckpt-epoch-{:0>5}'.format(epoch))
        # torch.save(state, './output/ckpt-epoch-{:0>5}'.format(epoch))

    print(''.join(['-'] * len(lin)))


def validate(model, dataloader, epoch=0):
    model.eval()
    pass


if __name__ == '__main__':

    dataset = Dataset('', size=args.img_dim)
    dataloader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # datasets = [Dataset('', size=sz) for sz in [256, 320, 416]]
    # dataloaders = [data.DataLoader(d, batch_size=bs, shuffle=True, num_workers=3) for d, bs in zip(datasets, [10, 16, 24])]

    model = DarkNet(args.model_cfg, cls_num=args.num_classes)
    # model = weight_init(model, path='./model/yolov3.pytorch')
    # model.load_state_dict(torch.load('yolov3.pytorch'), strict=False)
    # model.load_weights('./model/yolov3.weights')
    # torch.save(model.state_dict(), 'yolov3.pytorch')
    model = model.to(torch.device(device))

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)

    for e in range(int((args.epoches + 3) / 1)):

        scheduler.step()

        # random.shuffle(dataloaders)
        # for i, dtloader in enumerate(dataloaders):
        #     train(model, dtloader, optimizer, epoch=e * 3 + i)

        train(model, dataloader, optimizer, epoch=e)
        # validate(model, dataloader, epoch=e)


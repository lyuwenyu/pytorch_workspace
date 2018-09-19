import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import random
import glob

from data.dataset import Dataset
from model.build_model import DarkNet

import logging
logger = logging.getLogger('train')

import argparse
device = 'cuda:0' if not torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='')
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--dim', type=int, default=320)
args = parser.parse_args()

def train(model, dataloader, optimizer, epoch=0):

    model.train()

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

        bboxes = get_target()

        loss, losses = model(images, bboxes)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lin = ', '.join(['{}: {:.4f}'.format(k, v) for k, v in losses.items()])
        print('epoch/iter: {:0>3}/{:0>3}, lr: {}, {}'.format(e, i, format(optimizer.param_groups[0]['lr'], '.0e'), lin))

    if e % 10 == 0:
        # state = {
        #     'epoch': e,
        #     'model': model.state_dict(),
        #     'optimizer': optimizer.state_dict(),
        # }
        torch.save(model.state_dict(), './output/ckpt-epoch-{:0>5}'.format(epoch))


def validate(model, dataloader, epoch=0):
    model.eval()
    pass


if __name__ == '__main__':

    dataset = Dataset('', size=320)
    dataloader = data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=5)

    # datasets = [Dataset('', size=sz) for sz in [256, 320, 416]]
    # dataloaders = [data.DataLoader(d, batch_size=bs, shuffle=True, num_workers=3) for d, bs in zip(datasets, [10, 16, 24])]

    model = DarkNet('./_model/yolov3.cfg', cls_num=20)
    # model.load_weights('./model/yolov3.weights')

    if args.resume:
        path = sorted(glob.glob('output/ckpt*'))[-1]
        model.load_state_dict(torch.load(path))

    model = model.to(torch.device(device))

    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.99)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 60, 80], gamma=0.1)

    for e in range(int(102 / 1)):

        scheduler.step()

        # random.shuffle(dataloaders)
        # for i, dtloader in enumerate(dataloaders):
        #     train(model, dtloader, optimizer, epoch=e * 3 + i)

        train(model, dataloader, optimizer, epoch=e)
        
        # validate(model, dataloader, epoch=e)


import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from data.dataset import Dataset
from model.build_model import DarkNet

import logging
logger = logging.getLogger('train')

import argparse
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


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

        loss = model(images, bboxes)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('iter: {:0>5}, lr: {:0.10f}, loss: {:.5f}'.format(i, optimizer.param_groups[0]['lr'], loss.item()))

    if e % 10 == 0:
        torch.save(model.state_dict(), './output/ckpt-epoch-{:0>5}'.format(epoch))


def validate(model, dataloader, epoch=0):
    model.eval()
    pass


if __name__ == '__main__':

    dataset = Dataset('', size=416)
    dataloader = data.DataLoader(dataset, batch_size=8, num_workers=3)

    model = DarkNet('./_model/yolov3.cfg', img_size=416)
    model.load_weights('./model/yolov3.weights')
    model = model.to(torch.device(device))

    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.95)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    for e in range(100):

        scheduler.step()
        train(model, dataloader, optimizer, epoch=e)
        validate(model, dataloader, epoch=e)

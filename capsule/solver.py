import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.datasets.mnist import MNIST
import torch.utils.data as data

import numpy as np
from collections import OrderedDict
import logging
import pickle
import os

from tensorboardX import SummaryWriter
import network
from dataset import Dataset
import config


class Solver(object):

    def __init__(self, model=None, criteria=None, epoches=10, device='cpu'):

        self.device = device
        self.model = network.CapsuleNet().to(device)
        self.criteria = network.CapsuleLoss().to(device)

        # self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.optimizer = optim.Adam(self.model.parameters())
        # self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)
        
        self.epoches = epoches
        self.current_epoch = 0

        self.logger = self.get_logger()
        self.writer = SummaryWriter()
        # self.writer.close()
        print(self.model)

    @staticmethod
    def get_logger():

        file_handler = logging.FileHandler( 'log.txt' )
        fmat = logging.Formatter('%(name)s %(asctime)s %(levelname)-4s: %(message)s')
        file_handler.setFormatter(fmat)
        logger = logging.getLogger('solver')
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
        print('set logger done...')

        return logger


    def train(self, data_loader):
        # print(self.current_epoch)
        self.model.train()
        self.current_epoch += 1
        
        for i, (x, y) in enumerate(data_loader):
            x.to(self.device)
            y.to(self.device)

            logits = self.model(x)
            loss = self.criteria(logits, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            acc = (torch.argmax(logits, dim=1) == torch.argmax(y, dim=1)).type(torch.float32).mean()

            lin = 'epoch: {:0>2}, i: {:0>5}, loss: {:.3}, accuracy: {:.3}'.format(self.current_epoch, i, loss.item(), acc.item())
            self.logger.info(lin)
            self.writer.add_scalar('/train/loss', loss.item())
            self.writer.add_scalar('/train/accuracy', acc.item())
            print(lin)


    def test(self, data_loader):
        self.model.eval()
        print('\n\n----test-------')

        n = 0
        num = 0
        for _, (x, y) in enumerate(data_loader):
            x.to(self.device)
            y.to(self.device)

            logits = self.model(x)
            num += (torch.argmax(logits, dim=1) == torch.argmax(y, dim=1)).sum().item()
            n += x.size()[0]

        acc = 1.*num / n
        
        self.logger.info('test/accuracy: {}'.format(acc))
        self.writer.add_scalar('/test/accuracy', acc)
        print('accuracy: {}'.format(acc))


    def run(self, train_data_loader=None, test_data_loader=None):

        # self.test(test_data_loader)

        for i in range(0, self.epoches):

            # self.scheduler.step()
            self.train(train_data_loader)
            self.test(test_data_loader)

            if i % 10 == 0:
                self.save(prefix='{:0>3}'.format(i))

        self.save(prefix='final')


    def save(self, prefix=''):

        state = {
            'params': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            # 'scheduler': self.scheduler.state_dict(),
            'solver': self.state_dict()
        }
        torch.save(state, prefix+'.pt')


    def restore(self, path):

        state = torch.load(path)
        self.optimizer.load_state_dict(state['optimizer'])
        self.model.load_state_dict(state['params'])
        self.load_state_dict(state['solver'])


    def state_dict(self, ):

        # print(self.__dict__.keys())epoches

        des = OrderedDict()

        for k in self.__dict__:
            if 'model' == k:
                continue 
            if 'logger' == k:
                continue
            if 'criteria' == k:
                continue
            if 'scheduler' == k:
                continue
            if 'optimizer' == k:
                continue
            des[k] = self.__dict__[k]    
            
        return des

    def load_state_dict(self, state):
        self.__dict__.update(state)

    def __getstate__(self, ):
        return self.state_dict
    
    def __setstate__(self, state):
        self.load_state_dict(state)
        # self.__dict__.update(state)



if __name__ == '__main__':


    solver = Solver(epoches=30)
    
    dataset = Dataset(train=True)
    train_data_loader = data.DataLoader(dataset, batch_size=config.batch_size)
    dataset = Dataset(train=False)
    test_data_loader = data.DataLoader(dataset, batch_size=config.batch_size)

    solver.run(train_data_loader, test_data_loader)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # test_data = torch.load('../data/processed/test.pt')
    # print(len(test_data[1]))
    # print(test_data[0][0])
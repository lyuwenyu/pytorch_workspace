import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.datasets.mnist import MNIST
import torch.utils.data as data
from torch.nn.parallel.data_parallel import data_parallel

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

    def __init__(self, model=None, criteria=None, epoches=10, outputs_dir=config.output_dir):

        self.outputs_dir = outputs_dir
        if not os.path.exists(outputs_dir):
            os.mkdir(outputs_dir)
            
        self.device = torch.device(config.device)
        self.device_ids = config.device_ids
        
        self.model = network.CapsuleNet().to(self.device)
        self.criteria = network.CapsuleLoss().to(self.device)

        # self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        # self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        self.optimizer = optim.Adam(self.model.parameters())

        self.epoches = epoches
        self.current_epoch = 0

        self.logger = self.get_logger()
        self.writer = SummaryWriter()

        print(self.model)
        print(network.param_num(self.model))

        import atexit
        def exit():
            self.writer.close()
        atexit.register(exit)


    def get_logger(self):
        file_handler = logging.FileHandler( os.path.join(self.outputs_dir, 'log.txt') )
        fmat = logging.Formatter('%(name)s %(asctime)s %(levelname)-4s: %(message)s')
        file_handler.setFormatter(fmat)
        logger = logging.getLogger('solver')
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
        print('set logger done...')
        return logger


    def train(self, data_loader, set_log=True):

        self.model.train()
        
        for i, (x, y) in enumerate(data_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            # logits = self.model(x)
            logits = data_parallel(self.model, x, device_ids=self.device_ids)
            loss = self.criteria(logits, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            acc = (torch.argmax(logits, dim=1) == torch.argmax(y, dim=1)).type(torch.float32).mean()

            lin = 'epoch: {:0>2}, i: {:0>5}, loss: {:.3}, accuracy: {:.3}'.format(self.current_epoch, i, loss.item(), acc.item())
            if i%config.log_step == 0:
                print(lin)

            if set_log:
                self.logger.info(lin)
                self.writer.add_scalar('/train/loss', loss.item(), global_step=self.current_epoch*config.batch_size+i)
                self.writer.add_scalar('/train/accuracy', acc.item(), global_step=self.current_epoch*config.batch_size+i)
            
        self.current_epoch += 1


    def test(self, data_loader, set_log=True):
        self.model.eval()
        print('\n\n------test-------\n\n')

        n, num = 0, 0
        for _, (x, y) in enumerate(data_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            logits = self.model(x)
            num += (torch.argmax(logits, dim=1) == torch.argmax(y, dim=1)).sum().item()
            n += x.size()[0]

        acc = 1.*num / n
        print('accuracy: {}'.format(acc))

        if set_log:
            self.logger.info('test/accuracy: {}'.format(acc))
            self.writer.add_scalar('/test/accuracy', acc, global_step=self.current_epoch)


    def run(self, train_data_loader=None, test_data_loader=None):

        for i in range(0, self.epoches):

            # self.scheduler.step()
            self.train(train_data_loader)
            self.test(test_data_loader)

            if i % config.save_step == 0:
                self.save('{:0>3}'.format(i))

        self.save('final')


    def save(self, prefix):

        state = {
            'params': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            # 'scheduler': self.scheduler.state_dict(),
            'solver': self.state_dict()
        }

        torch.save(state, os.path.join(self.outputs_dir, prefix+'.pth'))


    def restore(self, path):

        state = torch.load(path)
        self.optimizer.load_state_dict(state['optimizer'])
        self.model.load_state_dict(state['params'])
        self.load_state_dict(state['solver'])


    def state_dict(self, ):

        des = OrderedDict()
        #
        # for k in self.__dict__:
        #     if 'model' == k:
        #         continue 
        #     if 'logger' == k:
        #         continue
        #     if 'criteria' == k:
        #         continue
        #     if 'scheduler' == k:
        #         continue
        #     if 'optimizer' == k:
        #         continue
        #     des[k] = self.__dict__[k]  
        # 
        des['current_epoch'] = self.__dict__['current_epoch']
        
        return des

    def load_state_dict(self, state):
        self.__dict__.update(state)

    def __getstate__(self, ):
        return self.state_dict
    
    def __setstate__(self, state):
        self.load_state_dict(state)
        # self.__dict__.update(state)



if __name__ == '__main__':


    solver = Solver(epoches=config.epochs)
    
    dataset = Dataset(train=True)
    train_data_loader = data.DataLoader(dataset, batch_size=config.batch_size)
    dataset = Dataset(train=False)
    test_data_loader = data.DataLoader(dataset, batch_size=config.batch_size)

    solver.run(train_data_loader, test_data_loader)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

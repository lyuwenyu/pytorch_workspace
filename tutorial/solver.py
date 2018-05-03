import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from collections import OrderedDict
import logging
import pickle
import os

class _Solver(object):
    pass



class Solver(object):

    def __init__(self, model=None, criteria=None, epoches=10):

        self.model = nn.Conv2d(10, 20, 3)
        self.criteria = criteria

        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)
        

        self.epoches = epoches
        self.current_epoch = 0

        self.logger = self.get_logger()


    def get_logger(self, ):

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
        self.logger.info(233)


    def test(self, data_loader):
        self.model.eval()

        self.logger.info('----')
        self.logger.info('test')
        self.logger.info('----')


    def run(self, train_data_loader=None, test_data_loader=None):

        for i in range(0, self.epoches):

            self.scheduler.step()

            self.train(train_data_loader)
            self.test(test_data_loader)

            if i % 3 == 0:
                self.save(prefix='{:0>3}'.format(i))

            print(i)

        self.save(prefix='final')


    def save(self, prefix=''):

        state = {
            'params': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
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

    solver = Solver()
    solver.run()


    print( solver.optimizer.state_dict() )
    solver.restore(path='003.pt')
    print( solver.optimizer.state_dict() )

    solver.run()
    print( solver.optimizer.state_dict() )

    # print(solver.current_epoch)
    # print(solver.__dict__)

    # with open('solver.pkl', 'wb') as f:
    #     pickle.dump(solver.state_dict(), f)

    # with open('solver.pkl', 'rb') as f:
    #     solver.load_state_dict( pickle.load(f) )

    # print(solver.current_epoch)

    # torch.save( solver.state_dict(), 'test' )
    # solver.load_state_dict( torch.load('test') )
    
    #print(solver.current_epoch)

    # print( pickle.dumps(solver) )
    # solver1 = pickle.loads( pickle.dumps(solver) )
    # print(solver.current_epoch)

    ###

    # m = nn.Conv2d(10, 20, 3)
    # optimizer = optim.SGD(m.parameters(), lr=0.1, momentum=0.9)

    # print( optimizer.state_dict() )
    
    # optimizer.load_state_dict( optimizer.state_dict() )

    # scheduler = lr_scheduler.StepLR(optimizer, step_size=5)
    
    # for i in range(20):
    #     scheduler.step()

    # print( optimizer.state_dict() )
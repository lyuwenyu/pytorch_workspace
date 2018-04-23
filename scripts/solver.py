import torch
import torch.nn as nn

from collections import OrderedDict
import logging
import pickle
import os

class Solver(object):

    def __init__(self, model=None, criteria=None, epoches=10):

        self.model = model
        self.criteria = criteria
        self.scheduler = ''

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
        self.current_epoch += 1
        self.logger.info(233)


    def test(self, data_loader):
        self.logger.info('----')
        self.logger.info('test')
        self.logger.info('----')


    def run(self, train_data_loader=None, test_data_loader=None):

        for i in range(self.epoches):

            self.train(train_data_loader)
            self.test(test_data_loader)

            if i % 5:
                self.save(prefix='{:0>3}'.format(i))
        
        self.save(prefix='final')


    def save(self, prefix=''):

        solver_path = '{}_solver'.format(prefix)
        # model_path = '{}_model'.format(prefix)
        # scheduler_path = '{}_scheduelr'.format(prefix)

        # torch.save(model_path, self.model.state_dict())
        
        with open(solver_path, 'wb') as f:
            pickle.dump(self.state_dict(), f)

        # scheduler
        


    def restore(self, prefix=''): # solver_path=None, model_path=None, scheduler_path=None):

        solver_path = '{}_solver'.format(prefix)
        # model_path = '{}_model'.format(prefix)
        # scheduler_path = '{}_scheduelr'.format(prefix)

        if os.path.exists(solver_path):
            with open(solver_path, 'rb') as f:
                solver.load_state_dict(pickle.load(f))

        # if os.path.exists(model_path):
        #     self.model.load_state_dict(torch.load(model_path))
            
        # if os.path.exists(scheduler_path):
        #     pass


    def state_dict(self, ):

        # print(self.__dict__.keys())

        des = OrderedDict()

        for k in self.__dict__:
            if 'model' == k:
                continue 
            if 'logger' == k:
                continue
            if 'criteria' == k:
                continue
            des[k] = self.__dict__[k]    
            
        return des

    def load_state_dict(self, state):
        self.__dict__.update(state)

    def __getstate__(self, ):
        return self.state_dict
    
    def __setstate__(self, state):
        self.load_state_dict(state)


if __name__ == '__main__':

    solver = Solver()

    solver.run()
    
    solver.restore('final')

    print(solver.current_epoch)

    # with open('solver.pkl', 'wb') as f:
    #     pickle.dump(solver.state_dict(), f)

    # with open('solver.pkl', 'rb') as f:
    #     solver.load_state_dict( pickle.load(f) )


    # print( pickle.dumps(solver) )
    # solver1 = pickle.loads( pickle.dumps(solver) )
    # print(solver.current_epoch)
"""
by. wenyu
reference: 
1. https://discuss.pytorch.org/t/torch-utils-data-dataset-random-split/32209/4
2. https://github.com/IgorSusmelj/pytorch-styleguide/issues/5#

"""

import torch
import torch.utils.data as data

from prefetch_generator import BackgroundGenerator


class DataloaderX(data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
    

class Dataset(DataloaderX):
    def __init__(self):
        pass
    
    def __len__(self, ):
        '''
        '''
        pass
    
    def __getitem__(self, i):
        '''
        '''
        pass

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from data.dataset import Dataset
from model.build_model import DarkNet

import logging

logger = logging.getLogger('train')


class Solver(object):
    
    def __init__(self, ):
        self.model = DarkNet('./_model/yolov3.cfg', img_size=416)
        self.model.load_weights('./model/yolov3.weights')

    def set_target(self, ):
        pass    
    
    def get_dataloader(self, ):
        pass

    def train(self, ):
        pass
    
    def inference(self, ):
        pass

    

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model = DarkNet('./_model/yolov3.cfg', img_size=416)
model.load_weights('./model/yolov3.weights')
model = model.to(torch.device(device))

opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.95)

dataset = Dataset('', size=416)
dataloader = data.DataLoader(dataset, batch_size=10, num_workers=3)


def train(model, epoch=0):

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

        opt.zero_grad()
        loss.backward()
        opt.step()

        print('iter: {:0>5}, lr: {:0.5f}, loss: {:.5f}'.format(i, 0, loss.item()))

        if i % 300 == 0:
            torch.save(model.state_dict(), './output/ckpt-{:0>5}'.format(i))
        

train(model)
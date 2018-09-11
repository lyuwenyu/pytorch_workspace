import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from data.dataset import Dataset
from model.build_model import DarkNet

import logging

logger = logging.getLogger('train')


class Solver(object):
    pass



model = DarkNet('./_model/yolov3.cfg', img_size=416)
model.load_weights('./model/yolov3.weights')

opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.95)

dataset = Dataset('', size=416)
dataloader = data.DataLoader(dataset, batch_size=3, num_workers=1)


model.train()
for i, (images, target) in enumerate(dataloader):

    def get_target():
        bboxes = []
        for ii in range(target.shape[0]):
            _target = target[ii]
            _target = _target[_target[:, 0] > 0]
            bboxes += [_target[:, 1:]]
        return bboxes 

    bboxes = get_target()

    print(images.shape, len(bboxes))

    loss = model(images, bboxes)

    opt.zero_grad()
    loss.backward()
    opt.step()

    print(loss)

    
    if i == 5:
        break
    



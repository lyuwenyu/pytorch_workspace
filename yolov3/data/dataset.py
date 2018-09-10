import torch
from torch.utils import data

from PIL import Image
import numpy as np
import random
import glob
import os
import Augmentor

import torchvision.transforms as transforms

from utils.ops_parse_xml import parse_xml
from utils.ops_pad_resize import pad_resize

from utils import ops_transform

class Dataset(data.Dataset):
    def __init__(self, path, size=512, yolo=True):
        
        root = '/home/wenyu/workspace/dataset/voc/VOCdevkit/VOC2007'
        self.img_root = os.path.join(root, 'JPEGImages')
        self.anns = glob.glob(os.path.join(root, 'Annotations', '*.xml'))

        self.totensor = transforms.ToTensor()

        self.bboxes_tensor = torch.ones(50, 5)

    def __len__(self):
        return len(self.anns)
    
    def __getitem__(self, i):
        
        blob = parse_xml(self.anns[i])

        filename = os.path.join(self.img_root, blob['filename'])

        img = Image.open(filename)
        bboxes = blob['bboxes']

        img, bboxes = pad_resize(img, bboxes)
        
        if random.random() < 0.5:
            img, bboxes = ops_transform.flip_lr(img, bboxes)

        if random.random() < 0.5:
            img, bboxes = ops_transform.flip_tb(img, bboxes)

        img, bboxes = ops_transform.pad_resize(img, bboxes, size=(600, 600))
        bboxes = ops_transform.xyxy2xywh(bboxes, img.size)

        img = self.totensor(img)
        self.bboxes_tensor.fill_(1)
        self.bboxes_tensor[:len(bboxes), 1:] = torch.from_numpy(bboxes)
        self.bboxes_tensor[len(bboxes): , 0] = 0

        return img, self.bboxes_tensor


if __name__ == '__main__':
    

    dataset = Dataset('')
    dataloader = data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=5) 
    
    for img, bboxes in dataloader:

        print(img.size(), bboxes.size())
        print(bboxes)

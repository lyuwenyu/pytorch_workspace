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
from utils.ops_show_bbox import show_bbox

from utils import ops_transform

class Dataset(data.Dataset):
    def __init__(self, path, size=512, num_classes=20):
        
        root = '/home/wenyu/workspace/dataset/voc/VOCdevkit/VOC2007'
        self.img_root = os.path.join(root, 'JPEGImages')
        self.anns = glob.glob(os.path.join(root, 'Annotations', '*.xml'))
        
        with open('./data/voc.names', 'r') as f:
            lines = f.readlines()
            lines = [ll.strip() for ll in lines]
            self.label_map = dict(zip(lines, range(len(lines))))
            # print(self.label_map)

        self.size = size

        self.totensor = transforms.ToTensor()

    def __len__(self):
        return len(self.anns)
    
    def __getitem__(self, i):
        
        blob = parse_xml(self.anns[i])
        filename = os.path.join(self.img_root, blob['filename'])
        bboxes = blob['bboxes']
        ngt = len(bboxes)
        label = [self.label_map[n] for n in blob['names']] 
        # assert len(label) == ngt, ''

        img = Image.open(filename)

        if random.random() < 0.5:
            img, bboxes = ops_transform.flip_lr(img, bboxes)

        if random.random() < 0.5:
            img, bboxes = ops_transform.flip_tb(img, bboxes)

        if random.random() < 0.5:
            img, bboxes = ops_transform.pad_resize(img, bboxes, size=(self.size, self.size))

        img, bboxes = ops_transform.resize(img, bboxes, size=(self.size, self.size))

        bboxes = ops_transform.xyxy2xywh(bboxes, img.size)     
        # mask cls bbx
        target_tensor = torch.zeros(50, 6)
        target_tensor[:ngt, 2: ] = torch.from_numpy(bboxes)
        target_tensor[:ngt, 1] = torch.tensor(label)
        target_tensor[:ngt, 0] = 1

        img = self.totensor(img)
        
        return img, target_tensor


if __name__ == '__main__':
    

    dataset = Dataset('')
    dataloader = data.DataLoader(dataset, batch_size=3, shuffle=True, num_workers=2) 
    
    topil = transforms.ToPILImage()

    for img, bboxes in dataloader:

        # print(img.size(), bboxes.size())

        _img = topil(img[0])
        _bboxes = bboxes[0]
        _bboxes = _bboxes[_bboxes[:, 0] > 0]
        _bbox = _bboxes[:, 2:]
        _classes = _bboxes[:, 1]

        # show_bbox(_img, ops_transform.xywh2xyxy(_bbox, _img.size), normalize=False)
        # show_bbox(_img, _bbox, xyxy=False, normalize=True)
        
        print(_classes)
        # break
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

        self.totensor = transforms.ToTensor()

    def __len__(self):
        return len(self.anns)
    
    def __getitem__(self, i):
        
        blob = parse_xml(self.anns[i])
        filename = os.path.join(self.img_root, blob['filename'])
        bboxes = blob['bboxes']
        ngt = len(bboxes)
        classes = blob['names']
        assert len(classes) == ngt, ''

        img = Image.open(filename)

        if random.random() < 0.5:
            img, bboxes = ops_transform.flip_lr(img, bboxes)

        if random.random() < 0.5:
            img, bboxes = ops_transform.flip_tb(img, bboxes)

        img, bboxes = ops_transform.pad_resize(img, bboxes, size=(512, 512))
        bboxes = ops_transform.xyxy2xywh(bboxes, img.size)

        img = self.totensor(img)

        bboxes_tensor = torch.zeros(50, 5)
        bboxes_tensor[:ngt, 1: ] = torch.from_numpy(bboxes)
        bboxes_tensor[:ngt, 0] = 1

        

        return img, bboxes_tensor


if __name__ == '__main__':
    

    dataset = Dataset('')
    dataloader = data.DataLoader(dataset, batch_size=3, shuffle=True, num_workers=2) 
    
    topil = transforms.ToPILImage()

    for img, bboxes in dataloader:

        _img = topil(img[2])
        _bbox = bboxes[2]
        _bbox = _bbox[_bbox[:, 0] > 0]
        _bbox = _bbox[:, 1:]

        show_bbox(_img, ops_transform.xywh2xyxy(_bbox, _img.size))



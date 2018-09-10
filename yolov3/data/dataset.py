import torch
from torch.utils import data

from PIL import Image
import numpy as np
import random
import glob
import os
import Augmentor

from utils.ops_parse_xml import parse_xml
from utils.ops_pad_resize import pad_resize
from utils.ops_transform import flip_lr, flip_tb, xyxy2xywh

class Dataset(data.Dataset):
    def __init__(self, path, size=512, yolo=True):
        
        root = '/home/wenyu/workspace/dataset/voc/VOCdevkit/VOC2007'
        self.img_root = os.path.join(root, 'JPEGImages')

        self.anns = glob.glob(os.path.join(root, 'Annotations', '*.xml'))


    def __len__(self):
        return len(self.anns)
    
    def __getitem__(self, i):
        
        blob = parse_xml(self.anns[i])

        filename = os.path.join(self.img_root, blob['filename'])

        img = Image.open(filename)
        bboxes = blob['bboxes']

        img, bboxes = pad_resize(img, bboxes)

        if random.random() < 0.5:
            img, bboxes = flip_lr(img, bboxes)

        if random.random() < 0.5:
            img, bboxes = flip_tb(img, bboxes)

        bboxes = xyxy2xywh(bboxes)

        print(bboxes)


        return img, bboxes


if __name__ == '__main__':
    

    dataset = Dataset('')
    dataloader = data.DataLoader(dataset, batch_size=3, shuffle=True, num_workers=3) 
    
    for item in dataloader:

        # print(item)

        break
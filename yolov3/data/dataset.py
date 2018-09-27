import torch
from torch.utils import data

from PIL import Image
import numpy as np
import random
import glob
import os
import sys
import Augmentor

import torchvision.transforms as transforms

from utils.ops_parse_xml import parse_xml
from utils.ops_pad_resize import pad_resize
from utils.ops_show_bbox import show_bbox
from utils.ops_show_bbox import show_tensor_bbox
from utils.ops_augmentor import ImagePipeline

from utils import ops_transform

class Dataset(data.Dataset):
    def __init__(self, annos_dir='', image_dir='', classes_path='', size=512, num_classes=20):
        
        self._set_voc_dataset()
        self._set_pipeline()

        self.size = size
        self.totensor = transforms.ToTensor()

    def _set_costum_dataset(self, ):
        ''' init
        self.anns
        self.image_dir
        self.label_map
        '''
        with open('./data/data.names', 'r') as f:
            lines = f.readlines()
            lines = [ll.strip() for ll in lines]
            self.label_map = dict(zip(lines, range(len(lines))))

        annos_dir = '/home/wenyu/workspace/dataset/'
        self.image_dir = annos_dir
        self.anns = glob.glob(os.path.join(annos_dir, '*.xml'))
        self.max_n = 50

    def _set_voc_dataset(self, ):
        ''' init
        self.anns
        self.image_dir
        self.label_map
        '''
        root = '/home/wenyu/workspace/dataset/voc/VOCdevkit/VOC2007'
        self.image_dir = os.path.join(root, 'JPEGImages')
        self.anns = glob.glob(os.path.join(root, 'Annotations', '*.xml'))
        self.max_n = 50
        with open('./data/voc.names', 'r') as f:
            lines = f.readlines()
            lines = [ll.strip() for ll in lines]
            self.label_map = dict(zip(lines, range(len(lines))))
    
    def _set_pipeline(self, ):
        '''augmentor'''
        self.pipeline = ImagePipeline()
        self.pipeline.random_brightness(probability=0.5, min_factor=0.5, max_factor=1.2)
        self.pipeline.random_distortion(probability=0.5, grid_height=20, grid_width=20, magnitude=1.0)
        self.pipeline.random_color(probability=0.5, min_factor=0.5, max_factor=1.5)
        self.pipeline.random_contrast(probability=0.5, min_factor=0.5, max_factor=1.5)

    def agumentation(self, img, bboxes, labels):
        '''_agumentation '''
        if random.random() < 0.5:
            img, bboxes = ops_transform.flip_lr(img, bboxes)

        # if random.random() < 0.5:
        #     img, bboxes = ops_transform.pad_resize(img, bboxes, size=(512, 512))
        # if random.random() < 0.5:
        #     img, bboxes, labels = ops_transform.random_perspective(img, bboxes, labels)

        if random.random() < 0.5:
            img, bboxes, labels = ops_transform.random_crop(img, bboxes, labels)

        img = self.pipeline.transform(img)

        return img, bboxes, labels

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, i):
        '''
        return:
        image: c, h, w (totensor)
        bboxes: n 6, [mask-0/1, label-0/cls, bbox-4/x/y/h/w]
        '''
        blob = parse_xml(self.anns[i])
        path = os.path.join(self.image_dir, blob['filename'])

        bboxes = np.array(blob['bboxes'])
        labels = np.array([self.label_map[n] for n in blob['names']])

        img = Image.open(path)

        if random.random() < 0.9:
            img, bboxes, labels = self.agumentation(img, bboxes, labels)

        if random.random() < 0.5:
            img, bboxes = ops_transform.pad_resize(img, bboxes, size=(self.size, self.size))
        else:
            img, bboxes = ops_transform.resize(img, bboxes, size=(self.size, self.size))

        # here convert bbox to yolo type
        bboxes = ops_transform.xyxy2xywh(bboxes, img.size)  

        if True: 
            '''shuffle label and bbox in one image,
            here to handle select same bbox in an image, 
            when more than one image choose same anchor.
            '''
            index = np.random.permutation(range(len(bboxes)))
            bboxes = bboxes[index]
            labels = labels[index]

        # mask cls bbx
        ngt = len(bboxes)
        target_tensor = torch.zeros(self.max_n, 6)
        target_tensor[:ngt, 2:] = torch.from_numpy(bboxes)
        target_tensor[:ngt, 1] = torch.tensor(labels)
        target_tensor[:ngt, 0] = 1

        # print(labels)
        img = self.totensor(img)
        
        return img, target_tensor

    
if __name__ == '__main__':
    

    dataset = Dataset('')
    dataloader = data.DataLoader(dataset, batch_size=100, shuffle=True, num_workers=2) 
    
    topil = transforms.ToPILImage()

    print(len(dataloader))

    for _ in range(100):
        for i, (img, bboxes) in enumerate(dataloader):

            if i in (10, 20, 30, 40):
                show_tensor_bbox(img[0], bboxes[0])
            
        break
    

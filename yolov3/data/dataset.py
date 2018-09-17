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
from utils.ops_show_bbox import show_tensor_bbox

from utils import ops_transform

class Dataset(data.Dataset):
    def __init__(self, annos_dir='', image_dir='', classes_path='', size=512, num_classes=20):
        

        self._set_costum_dataset()

        self.size = size
        self.totensor = transforms.ToTensor()

    def _set_costum_dataset(self, ):
        ''' init
        self.anns
        self.image_dir
        self.label_map
        '''
        with open('./data/fisheye.names', 'r') as f:
            lines = f.readlines()
            lines = [ll.strip() for ll in lines]
            self.label_map = dict(zip(lines, range(len(lines))))

        annos_dir = '/home/wenyu/workspace/dataset/fisheye/tc_20180816'
        self.image_dir = annos_dir
        self.anns = glob.glob(os.path.join(annos_dir, '*.xml'))
        self.max_n = 50

    def _set_voc_dataset(self, ):
        ''' init
        self.anns
        self.image_dir
        self.label_map
        '''
        # root = '/home/wenyu/workspace/dataset/voc/VOCdevkit/VOC2007'
        # self.img_root = os.path.join(root, 'JPEGImages')
        # self.anns = glob.glob(os.path.join(root, 'Annotations', '*.xml'))
        # with open('./data/voc.names', 'r') as f:


    def __len__(self):
        return len(self.anns)
    

    def _agumentation(self, img, bboxes, labels):
        '''_agumentation '''
        if random.random() < 0.5:
            img, bboxes = ops_transform.flip_lr(img, bboxes)

        # if random.random() < 0.1:
        #     img, bboxes = ops_transform.flip_tb(img, bboxes)
        
        if random.random() < 0.2:
            img, bboxes = ops_transform.pad_resize(img, bboxes)

        if random.random() < 0.5:
            img, bboxes, labels = ops_transform.random_perspective(img, bboxes, labels)

        if random.random() < 0.5:
            img, bboxes, labels = ops_transform.random_crop(img, bboxes, labels)

        return img, bboxes, labels


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

        if random.random() < 0.8:
            img, bboxes, labels = self._agumentation(img, bboxes, labels)

        img, bboxes = ops_transform.resize(img, bboxes, size=(self.size, self.size))

        # here to yolo bbox type
        bboxes = ops_transform.xyxy2xywh(bboxes, img.size)  

        # mask cls bbx
        ngt = len(bboxes)
        target_tensor = torch.zeros(self.max_n, 6)
        target_tensor[:ngt, 2: ] = torch.from_numpy(bboxes)
        target_tensor[:ngt, 1] = torch.tensor(labels)
        target_tensor[:ngt, 0] = 1

        img = self.totensor(img)
        
        return img, target_tensor

    
if __name__ == '__main__':
    

    dataset = Dataset('')
    dataloader = data.DataLoader(dataset, batch_size=3, shuffle=True, num_workers=2) 
    
    topil = transforms.ToPILImage()

    for img, bboxes in dataloader:

        show_tensor_bbox(img[0], bboxes[0])

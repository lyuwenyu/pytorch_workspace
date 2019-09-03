import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import glob
import os
import sys
sys.path.insert(0, '/home/wenyu/workspace/pytorch_workspace')
from augmentor.utils.ops_pipeline import ImagesPipeline
import random
from yolov3.utils import ops_transform
from yolov3.utils import ops_parse_xml
from yolov3.utils import ops_show_bbox

from PIL import Image

class Dataset(data.Dataset):
    def __init__(self, annos_dir='', image_dir='', classes_path='', size=512,):
        
        self._set_voc_dataset()
        self._set_pipeline()
        # self.attr_num = num_classes + 4

        self.size = size
        self.totensor = transforms.ToTensor()

    def _set_voc_dataset(self, ):
        ''' init
        self.anns
        self.image_dir
        self.label_map
        '''
        root = '../../dataset/voc/VOCdevkit/VOC2007'
        self.image_dir = os.path.join(root, 'JPEGImages')
        self.anns = glob.glob(os.path.join(root, 'Annotations', '*.xml'))
        self.imgs = [path.replace('Annotations', 'JPEGImages').replace('xml', 'jpg') for path in self.anns]
        self.max_n = 50
        with open('./data/voc.names', 'r') as f:
            lines = f.readlines()
            lines = [ll.strip() for ll in lines]
            self.label_map = dict(zip(lines, range(1, 1 + len(lines))))


    def _set_pipeline(self, ):
        '''augmentor'''
        self.pipeline = ImagesPipeline() 
        self.pipeline.random_brightness(probability=0.5, min_factor=0.5, max_factor=1.2)
        self.pipeline.random_distortion(probability=0.5, grid_height=20, grid_width=20, magnitude=1.0)
        self.pipeline.random_color(probability=0.5, min_factor=0.5, max_factor=1.5)
        self.pipeline.random_contrast(probability=0.5, min_factor=0.5, max_factor=1.5)

    def agumentation(self, img, bboxes, labels):
        '''_agumentation '''

        if random.random() < 0.5:
            img, bboxes = ops_transform.flip_lr(img, bboxes)

        if random.random() < 0.5:
            img, bboxes, labels = ops_transform.random_crop(img, bboxes, labels)

        img = self.pipeline.transform(img)[0]

        return img, bboxes, labels


    def __len__(self):
        return len(self.anns)


    def __getitem__(self, i):
        '''
        return:
        image: c, h, w (totensor)
        bboxes: n 6, [mask-0/1, label-0/cls, bbox-4/x/y/h/w]
        '''
        
        # path = os.path.join(self.image_dir, blob['filename'])

        bboxes, labels = [], []

        blob = ops_parse_xml.parse_xml(self.anns[i])
        path = self.imgs[i]

        for bbx, label in zip(blob['bboxes'], blob['names']):
            if label in self.label_map:
                bboxes += [bbx]
                labels += [self.label_map[label]]
        bboxes = np.array(bboxes)
        labels = np.array(labels)

        if len(labels) == 0:
            return self[(i+1)%len(self)]

        # need below data
        img = Image.open(path)

        if random.random() < 0.9:
            img, bboxes, labels = self.agumentation(img, bboxes, labels)

        if random.random() < 0.5:
            img, bboxes = ops_transform.pad_resize(img, bboxes, size=(self.size, self.size))
        else:
            img, bboxes = ops_transform.resize(img, bboxes, size=(self.size, self.size))

        # here convert bbox to yolo type
        # bboxes = ops_transform.xyxy2xywh(bboxes, img.size) 
        target_tensor = torch.zeros(self.max_n, 6)

        if True: 
            '''shuffle label and bbox in one image,
            here to handle select same bbox in an image, 
            when more than one image choose same anchor.
            '''
            index = np.random.permutation(range(len(bboxes)))
            bboxes = bboxes[index]
            labels = labels[index]

        # ops_show_bbox.show_bbox(img, bboxes)
        # mask cls bbx
        ngt = len(bboxes)

        target_tensor[:ngt, 2:] = torch.from_numpy(bboxes) / self.size # normalize coor
        target_tensor[:ngt, 1] = torch.tensor(labels) # from 1, bg is 0
        target_tensor[:ngt, 0] = 1
        
        
        # print(labels)
        img = self.totensor(img)
        
        return img, target_tensor


if __name__ == '__main__':
    
    dataset = Dataset()
    dataloader = data.DataLoader(dataset, batch_size=5, shuffle=True)

    for i, (imgs, targets) in enumerate(dataloader):

        print(i)
        print(imgs.shape, targets.shape)
        ops_show_bbox.show_tensor_bbox(imgs[0], targets[0], xyxy=True, normalized=True)
        break
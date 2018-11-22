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

sys.path.insert(0, '/home/wenyu/workspace/pytorch_workspace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.ops_parse_xml import parse_xml
from utils.ops_pad_resize import pad_resize
from utils.ops_show_bbox import show_bbox, show_polygon
from utils.ops_show_bbox import show_tensor_bbox
from utils.ops_augmentor import ImagePipeline
from utils.ops_parse_json import load_json
from utils import ops_transform

# from augmentor.utils import pipeline

class Dataset(data.Dataset):
    def __init__(self, annos_dir='', image_dir='', classes_path='', size=512, num_classes=20, quad=False):
        
        # self._set_voc_dataset()
        self._set_fisheye_dataset()
        self._set_pipeline()
        
        self.quad = quad
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
    
    def _set_fisheye_dataset(self, ):
        with open('./data/fisheye.names', 'r') as f:
            lines = f.readlines()
            lines = [ll.strip() for ll in lines]
            self.label_map = dict(zip(lines, range(len(lines))))
        # print(self.label_map)

        annos_dir = '/'
        anns = glob.glob(os.path.join(annos_dir, '', '*', 'anno', '*.xml'))
        anns += glob.glob(os.path.join(annos_dir, '', '*','anno', '*.xml'))
        imgs = [ann.replace('xml', 'jpg').replace('anno', 'img') for ann in anns]

        assert len(anns) == len(imgs), ''
        self.imgs, self.anns = [], []
        for ann, img in zip(anns, imgs):
            if os.path.exists(ann) and os.path.exists(img):
                self.imgs += [img]
                self.anns += [ann]

        self.max_n = 50


    def _set_pipeline(self, ):
        '''augmentor'''
        self.pipeline = ImagePipeline()
        self.pipeline.random_brightness(probability=0.5, min_factor=0.5, max_factor=1.2)
        self.pipeline.random_distortion(probability=0.5, grid_height=20, grid_width=20, magnitude=1.0)
        self.pipeline.random_color(probability=0.5, min_factor=0.5, max_factor=1.5)
        self.pipeline.random_contrast(probability=0.5, min_factor=0.5, max_factor=1.5)

    def agumentation(self, img, bboxes, labels):
        '''_agumentation '''
        # if random.random() < 0.5:
        #     img, bboxes = ops_transform.flip_lr(img, bboxes)

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
        
        # path = os.path.join(self.image_dir, blob['filename'])
        
        bboxes, labels = [], []
        while len(labels) == 0:
            bboxes, labels = [], []

            blob = parse_xml(self.anns[i])
            path = self.imgs[i]

            for bbx, label in zip(blob['bboxes'], blob['names']):
                if label in self.label_map:
                    bboxes += [bbx]
                    labels += [self.label_map[label]]
            bboxes = np.array(bboxes)
            labels = np.array(labels)

            i = (i+1) % len(self.imgs)

        # need below data
        img = Image.open(path)
        # bboxes = 
        # labels = 

        if random.random() < 0.9:
            img, bboxes, labels = self.agumentation(img, bboxes, labels)

        if random.random() < 0.5:
            img, bboxes = ops_transform.pad_resize(img, bboxes, size=(self.size, self.size))
        else:
            img, bboxes = ops_transform.resize(img, bboxes, size=(self.size, self.size))

        # here convert bbox to yolo type
        bboxes = ops_transform.xyxy2xywh(bboxes, img.size) 

        if self.quad:
            # bboxes = [[x-w/2, y-h/2, x+w/2, y-h/2, x+w/2, y+h/2, x-w/2, y+h/2] for (x, y, w, h) in bboxes]
            # bboxes = np.array(bboxes)
            target_tensor = torch.zeros(self.max_n, 10)
        else:
            target_tensor = torch.zeros(self.max_n, 6)

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
        # target_tensor = torch.zeros(self.max_n, 6)
        # target_tensor = torch.zeros(self.max_n, 10)

        target_tensor[:ngt, 2:] = torch.from_numpy(bboxes)
        target_tensor[:ngt, 1] = torch.tensor(labels)
        target_tensor[:ngt, 0] = 1

        # print(labels)
        img = self.totensor(img)
        
        return img, target_tensor


class DatasetQuad(data.Dataset):

    def __init__(self, root='', img_dim=416, ):

        self.root = ''
        self.annos = glob.glob(os.path.join(self.root, '*.json'))
        self.max_n = 20
        self.img_dim = img_dim

        self.totensor = transforms.ToTensor()
        
        self.pipeline = ImagePipeline()
        self.pipeline.random_brightness(probability=0.5, min_factor=0.5, max_factor=1.5)
        self.pipeline.random_distortion(probability=0.5, grid_height=10, grid_width=10, magnitude=1.0)
        self.pipeline.random_contrast(probability=0.5, min_factor=0.5, max_factor=1.5)

    def __len__(self, ):
        return len(self.annos)

    def __getitem__(self, i):
        
        anno = self.annos[i]
        blob = load_json(anno)

        img = blob['imageData']
        newimg = self.pipeline.transform(img)
        # show_polygon(blob['imageData'], blob['points'])
        newimg = newimg.resize((self.img_dim, self.img_dim))
        points = torch.from_numpy(np.array(blob['points']).astype(np.float32)) * (self.img_dim / img.size[0])
        points /= self.img_dim

        ngt = len(blob['labels'])
        
        if ngt == 0:
            print(anno)
            return self[(i + 1) % len(self)]

        target_tensor = torch.zeros(self.max_n, 10)
        target_tensor[: ngt, 2:] = points
        target_tensor[: ngt, 1] = 0 # label-0, one class
        target_tensor[: ngt, 0] = 1

        newimg = self.totensor(newimg)

        return newimg, target_tensor


if __name__ == '__main__':
    

    # dataset = Dataset()
    # dataloader = data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=8) 
    
    dataset = DatasetQuad()
    dataloader = data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2) 
    
    topil = transforms.ToPILImage()

    print(len(dataloader))

    for _ in range(1):
        for i, (img, bboxes) in enumerate(dataloader):
            if i in (10, 20, 30,):
                bbox = bboxes[0].data.numpy()
                bbox = bbox[bbox[:, 0] == 1]
                bbox = bbox[:, 2:]
                print(bbox.shape)
                show_polygon(topil(img[0]), bbox, normalized=True)
                print

                xmin = np.min(bbox[:, [0, 2, 4, 6]], axis=1).reshape(-1, 1)
                xmax = np.max(bbox[:, [0, 2, 4, 6]], axis=1).reshape(-1, 1)
                ymin = np.min(bbox[:, [1, 3, 5, 7]], axis=1).reshape(-1, 1)
                ymax = np.max(bbox[:, [1, 3, 5, 7]], axis=1).reshape(-1, 1)
                
                bbox = np.concatenate((xmin, ymin, xmax, ymax), axis=1)
                # print(bbox)
                gx = (xmin + xmax) / 2.
                gy = (ymin + ymax) / 2.
                gw = xmax - xmin
                gh = ymax - ymin

                bbox = np.concatenate((gx, gy, gw, gh), axis=1)

                show_bbox(topil(img[0]) , bbox, normalized=True, xyxy=False)
                # c += 1
                
            # print(img.shape)
            # print(bboxes.shape)

        break
    
    

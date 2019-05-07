import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from PIL import ImageDraw

import os
import glob
import math
import time
import numpy as np

from . import utils
from . import vutils

class DetDataset(data.Dataset):
    def __init__(self, cfg, parser='', augmentor=''):
        
        anno_root = cfg['anno_root']
        anno_ext = cfg['anno_ext']
        
        self.cfg = cfg
        self.anns = glob.glob(os.path.join(anno_root, '*.' + anno_ext))
        self.parser = parser
        self.augmentor = augmentor

    def __len__(self, ):
        '''
        '''
        return len(self.anns)


    def __getitem__(self, ii):
        '''
        '''
        
        blob = self.parser(self.anns[ii])
        
        image = blob['imageData']
        labels = blob['labels']
        points = blob['points'] / [blob['width'], blob['height']]
        bboxes = blob['bboxes'] / ([blob['width'], blob['height']] * 2)

        # augmentor
        # anns = {'image': image, 'bboxes': bboxes, 'labels': labels}
        # result = self.augmentor(**anns)
        # image = np.array(result['image'])
        # bboxes = np.array(result['bboxes'])
        # labels = np.array(result['labels'])
        image = Image.fromarray(image).resize((640, 640))
        image = np.array(image)
        
        if len(labels) == 0:
            print('---------')
            return self[(ii+1)%len(self)]

        # target
        ih, iw = image.shape[:-1]
        oh, ow = ih // self.cfg['stride'], iw // self.cfg['stride']
        hm = np.zeros((oh, ow, self.cfg['num_classes']), dtype=np.float32)
        
        wh = np.zeros((oh, ow, 2), dtype=np.float32)
        off = np.zeros((oh, ow, 2), dtype=np.float32)
        quad = np.zeros((oh, ow, 8), dtype=np.float32)
        mask = np.zeros((oh, ow), dtype=np.uint8)

        for k in range(len(labels)):
            bbx = bboxes[k] * ([iw, ih] * 2) / self.cfg['stride']
            pts = points[k] * [iw, ih] / self.cfg['stride']

            lab = 0 # int(labels[k])
            w, h = bbx[2] - bbx[0], bbx[3] - bbx[1]
            cx, cy = (bbx[2] + bbx[0]) / 2, (bbx[3] + bbx[1]) / 2
            ci, cj = int(cx), int(cy)

            radius = max(0, int(utils.gaussian_radius((math.ceil(h), math.ceil(w)))))
            utils.draw_umich_gaussian(hm[:, :, lab], (ci, cj), radius)

            wh[cj, ci, :] = w, h
            off[cj, ci, :] = cx - ci, cy - cj
            quad[cj, ci, :] = (pts - [cx, cy]).reshape(8)
            mask[cj, ci] = 1

        if self.cfg['debug']:
            name = str(time.time())
            self.show_bbox(image, bboxes, points, name)
            self.show_gaussian(hm, name)

        image = self.totensor(image)

        target = {'image': image, 'hm': hm, 'quad': quad, 'wh': wh, 'off': off, 'mask': mask}

        return target
   

    def totensor(self, image, normalize=True):
        '''
        ''' 
        totensor = transforms.ToTensor()
        image = totensor(image)
        
        if normalize:
            image= transforms.Normalize(mean=self.cfg['mean'], std=self.cfg['std'])(image)
            pass
        return image

    
    def show_gaussian(self, heatmap, name=''):
        '''n x h x w
        '''
        _im = np.max(heatmap, axis=-1)
        _im = np.floor(_im * 255)
        _im = Image.fromarray(_im).convert('L')
        _im.save(f'./tmp/{name}_heatmap.jpg')


    def show_bbox(self, image, bboxes, quad, name=''):
        '''
        '''
        _img = Image.fromarray(image)
        _w, _h = _img.size
        _draw = ImageDraw.Draw(_img)
        for i in range(len(bboxes)):
            _draw.rectangle(tuple(bboxes[i] * ([_w, _h]*2)), outline='red')
            _draw.polygon(tuple((quad[i] * [_w, _h]).reshape(-1)), outline='yellow')

        _img.save(f'./tmp/{name}_bbox.jpg')


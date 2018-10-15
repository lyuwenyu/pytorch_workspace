import json
import numpy as np
import os
import sys
from PIL import Image, ImageDraw
import torch.utils.data as data
import torchvision.transforms as transforms
import glob

sys.path.insert(0, '/home/wenyu/workspace/pytorch_workspace/yolov3/utils')
from ops_load_json import load_json

from .ops_label_color import label_color, label_color_map

blob = load_json('./data/00001.json')
img = Image.open('./data/00001.jpg')
# print(blob)]
img = img.resize([512, 512])

anno = Image.new('RGB', img.size, color=label_color['background'])
anno_draw = ImageDraw.Draw(anno)

for lab, pts in zip(blob['label'], blob['points']):
    # print(lab, pts)
    anno_draw.polygon(pts, fill=label_color[lab])

anno.resize([512, 512])

ann_arr = np.array(anno)
ann_msk = np.zeros(ann_arr.shape[:-1], dtype=np.uint8)

for k, v in label_color_map.items():
    msk = np.all(ann_arr == np.array(k).reshape(1, 1, 3), axis=2)
    ann_msk[msk] = v * 100

print(ann_msk.shape)

mask = Image.fromarray(ann_msk)
mask.show()
anno.show()


class Dataset(data.Dataset):
    def __init__(self, data_dir, img_size=512):
        self.data_dir = data_dir
        self.jsons = glob.glob(os.path.join(data_dir, '*.json'))
        self.img_size = img_size

        self.totensor = transforms.ToTensor()

    def __len__(self, ):
        return len(self.jsons)


    def __getitem__(self, i):
        
        blob = load_json(self.jsons[i])

        img = Image.open(os.path.join(self.data_dir, blob['imagePath']))
        img = img.resize([self.img_size, self.img_size])

        # mask label
        anno = Image.new('RGB', img.size, color=label_color['background'])
        anno_draw = ImageDraw.Draw(anno)
        for lab, pts in zip(blob['label'], blob['points']):
            anno_draw.polygon(pts, fill=label_color[lab])
        anno = anno.resize([self.img_size, self.img_size])

        ann_arr = np.array(anno)
        ann_msk = np.zeros(ann_arr.shape[:-1], dtype=np.uint8)

        for k, v in label_color_map.items():
            msk = np.all(ann_arr == np.array(k).reshape(1, 1, 3), axis=2)
            ann_msk[msk] = v

        return self.totensor(img), ann_msk

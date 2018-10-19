import json
import numpy as np
import os
import sys
from PIL import Image, ImageDraw
import torch.utils.data as data
import torchvision.transforms as transforms
import glob

sys.path.insert(0, '/home/wenyu/workspace/pytorch_workspace/')
# sys.path.insert(0, '/home/wenyu/workspace/pytorch_workspace/')
import yolov3.utils.ops_parse_json as ops_parse_json
import augmentor.utils.pipeline as pipeline

from data.ops_label_color import label_color, label_color_map

blob = ops_parse_json.load_json('./data/00001.json')
img = Image.open('./data/00001.jpg')
# print(blob)]
print(blob)

anno = Image.new('RGB', img.size, color=label_color['background'])
anno_draw = ImageDraw.Draw(anno)

for lab, pts in zip(blob['labels'], blob['points']):
    anno_draw.polygon(pts, fill=label_color[lab])

img = img.resize([512, 512])
anno = anno.resize([512, 512])

ann_arr = np.array(anno)
ann_msk = np.zeros(ann_arr.shape[:-1], dtype=np.uint8)

for k, v in label_color_map.items():
    msk = np.all(ann_arr == np.array(k).reshape(1, 1, 3), axis=2)
    ann_msk[msk] = v * 100

print(ann_msk.shape)

mask = Image.fromarray(ann_msk)
# mask.show()
# anno.show()


class Dataset(data.Dataset):
    def __init__(self, data_dir, img_size=512):
        self.data_dir = data_dir
        self.jsons = glob.glob(os.path.join(data_dir, '*.json'))
        self.img_size = img_size

        self.totensor = transforms.ToTensor()

        self.simu_pipe = pipeline.ImagesPipeline()
        self.simu_pipe.flip_left_right(probability=0.5)
        # self.simu_pipe.crop_by_size(probability=0.8, width=416, height=416)
        self.simu_pipe.crop_random(probability=0.7, percentage_area=0.7)
        # self.simu_pipe.resize(probability=1.0, width=img_size, height=img_size) # HERE do not do it in here

        self.img_pipe = pipeline.ImagesPipeline()
        self.img_pipe.random_brightness(probability=0.5, min_factor=0.8, max_factor=1.5)
        self.img_pipe.random_contrast(probability=0.5, min_factor=0.8, max_factor=2.0)

    def __len__(self, ):
        return len(self.jsons)


    def __getitem__(self, i):
        
        blob = ops_parse_json.load_json(self.jsons[i])
        img = Image.open(os.path.join(self.data_dir, blob['imagePath']))

        # mask label
        anno = Image.new('RGB', img.size, color=label_color['background'])
        anno_draw = ImageDraw.Draw(anno)
        for lab, pts in zip(blob['labels'], blob['points']):
            anno_draw.polygon(pts, fill=label_color[lab])

        # some augmentation for image anno
        img, anno = self.simu_pipe.transform([img, anno])
        img = self.img_pipe.transform([img])[0]

        img = img.resize((self.img_size, self.img_size))
        anno = anno.resize((self.img_size, self.img_size))

        # final msk and img
        ann_arr = np.array(anno)
        ann_msk = np.zeros(ann_arr.shape[:-1], dtype=np.uint8)
        for k, v in label_color_map.items():
            msk = np.all(ann_arr == np.array(k).reshape(1, 1, 3), axis=2)
            ann_msk[msk] = v

        if True:
            img.show()
            anno.show()
            # Image.fromarray(ann_msk * 50).show()
            c += 1

        return self.totensor(img), ann_msk


    def augmentation(self, ):
        pass

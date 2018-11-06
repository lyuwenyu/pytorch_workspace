import numpy as np
import time
from PIL import Image, ImageDraw

import torch
from torchvision import transforms
import torch.nn.functional as F
# from model.build_ssd_model import SSD
from model.vgg_ssd300 import build_ssd
import os, sys

sys.path.insert(0, '/home/wenyu/workspace/pytorch_workspace')
from yolov3.utils.ops_nms import NMS
from yolov3.utils.ops_show_bbox import show_bbox
from yolov3.utils import ops_transform

import argparse
import glob
import random


parser = argparse.ArgumentParser()
parser.add_argument('--path', default='./model/001.png')
parser.add_argument('--dim', type=int, default=300)
args = parser.parse_args()

def pre_image(img, inp_dim):
    totensor = transforms.ToTensor()
    img = ops_transform.resize(Image.open(img), size=(inp_dim, inp_dim))
    return totensor(img).unsqueeze(0)

dim = 300
# dim = 256 
# dim = 224
# dim = args.dim

device = 'cuda:0' if not torch.cuda.is_available() else 'cpu'
device = torch.device(device)

model = build_ssd(size=300, num_classes=21)
model.eval()
model = model.to(torch.device(device))
model.load_state_dict(torch.load('./output/state-ckpt-epoch-00005', map_location='cpu')['model'])
# model.load_weights('./model/yolov3.weights')
print('---load---done-----')

# print(model)

paths = glob.glob('../../dataset/voc/VOCdevkit/VOC2007/JPEGImages/*.jpg')
random.shuffle(paths)
args.path = paths[0]

image = pre_image(args.path, dim)
data = image.to(dtype=torch.float32, device=device)

tic = time.time()
conf, loc = model(data.to(device=device))
print('time: ', time.time()-tic)
print('conf: ', conf.shape)
print('loc: ', loc.shape)

bbox = loc[0].cpu().data.numpy()
prob = conf[0].cpu().data.numpy()

labs = prob.argmax(axis=-1)

bbox = bbox[labs > 0]
prob = prob[labs > 0]


if len(bbox) > 0:
    show_bbox(ops_transform.resize(Image.open(args.path), size=(dim, dim)), bbox, xyxy=True, normalized=True)

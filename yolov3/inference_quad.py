import numpy as np
import time
from PIL import Image, ImageDraw

import torch
from torchvision import transforms

from model.build_model import DarkNet
from utils.ops_nms import NMS
from utils.ops_show_bbox import show_bbox, show_polygon
from utils import ops_transform

import argparse
import glob
import random


parser = argparse.ArgumentParser()
parser.add_argument('--path', default='./_model/dog-cycle-car.png')
parser.add_argument('--dim', type=int, default=416)
args = parser.parse_args()

def pre_image(img, inp_dim):
    totensor = transforms.ToTensor()
    img = ops_transform.resize(Image.open(img), size=(inp_dim, inp_dim))
    return totensor(img).unsqueeze(0)

dim = 416
# dim = 256 
# dim = 224
# dim = args.dim

device = 'cuda:0' if not torch.cuda.is_available() else 'cpu'
# pylint: disable=E1101
device = torch.device(device)
# pylint: enable=E1101

model = DarkNet('./_model/yolov3.cfg', cls_num=1, quad=True)
model.load_state_dict(torch.load('./output/state-ckpt-epoch-00120', map_location='cpu')['model'])

model.eval()
model = model.to(device=device)

paths = glob.glob('*.jpg')
random.shuffle(paths)
args.path = paths[0]
print(args.path)

image = pre_image(args.path, dim)
data = image.to(dtype=torch.float32, device=device)


tic = time.time()
predx = model(data.to(device=device))
print('time: ', time.time()-tic)

pred = predx[0]

objectness_threshold = 0.99
if len(pred[pred[:, 12] > objectness_threshold]) > 0:
    pred = pred[pred[:, 12] > objectness_threshold].cpu().data.numpy()
    img = ops_transform.resize(Image.open(args.path), size=(dim, dim))
    show_bbox(img, pred[:, 0: 4], xyxy=False, normalized=False)
    
    img = ops_transform.resize(Image.open(args.path), size=(dim, dim))
    draw = ImageDraw.Draw(img)
    for polyg in pred:
        if all(list(map(lambda x: 0 <= x < dim, polyg))):
            draw.polygon(tuple(polyg[4: 12]), outline='red')
    img.show()

else:
    print(f'--{objectness_threshold}-no objs---')


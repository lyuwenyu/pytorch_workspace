import numpy as np
import time
from PIL import Image, ImageDraw

import torch
from torchvision import transforms

from model.build_model import DarkNet
from utils.ops_nms import NMS
from utils.ops_show_bbox import show_bbox
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
    img = ops_transform.pad_resize(Image.open(img), size=(inp_dim, inp_dim))
    return totensor(img).unsqueeze(0)

# dim = 608
# dim = 448
# dim = 416
dim = 320
# dim = 256 
# dim = 224
# dim = args.dim

device = 'cpu' # cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

model = DarkNet('./_model/yolov3.cfg', cls_num=20)
# model.load_state_dict(torch.load('yolov3.pytorch'))
model.load_state_dict(torch.load('./output/ckpt-epoch-00060'))
# model.load_weights('./model/yolov3.weights')

model.eval()
model = model.to(device=device)
# print(model)

paths = glob.glob('/home/wenyu/workspace/dataset/voc/VOCdevkit/VOC2007/JPEGImages/*.jpg')
random.shuffle(paths)
args.path = paths[0]

image = pre_image(args.path, dim)
data = image.to(dtype=torch.float32, device=device)

tic = time.time()
predx = model(data.to(device=device))
print('time: ', time.time()-tic)

pred = predx[0]

# objectness_threshold = 0.99
# if len(pred[pred[:, 4] > objectness_threshold]) > 0:
#     show_bbox(ops_transform.pad_resize(Image.open(args.path), size=(dim, dim)), pred[pred[:, 4] > objectness_threshold].cpu().data.numpy()[:, 0: 4], xyxy=False, normalized=False)
# else:
#     print(f'--{objectness_threshold}-no objs---')

pred = pred.cpu().data.numpy()
result = NMS(pred, objectness_threshold=0.95, class_threshold=0.7, iou_threshold=0.4)

for k, v in result.items():
    show_bbox(ops_transform.pad_resize(Image.open(args.path), size=(dim, dim)), v[0])

    print('k: ', k)
    print('b: ', v[0])
    print('p: ', v[1])
    print()

    
# print('++++++++++++++++++++++++++++++++++++++++')
# import torch.onnx
# import caffe2.python.onnx.backend as backbend
# import onnx

# dummy_input = torch.zeros(1, 3, args.dim, args.dim)
# torch.onnx.export(model, dummy_input, 'test.onnx', verbose=True)
# model = onnx.load('test.onnx')
# onnx.checker.check_model(model)
# # onnx.helper.printable_graph(model.graph) 

# rep = backbend.prepare(model, device='CPU')

# outs = rep.run(image.cpu().data.numpy().astype(np.float32))
# outs = outs[0][0]

# print(type(outs))
# print(outs[0].shape)

# show_bbox(Image.open(args.path).resize((dim, dim)), outs[outs[:, 0] > 0.3][:, 1:5], xyxy=False, normalized=False)

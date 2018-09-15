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

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='./_model/dog-cycle-car.png')
parser.add_argument('--dim', type=int, default=416)
args = parser.parse_args()

def pre_image(img, inp_dim):
    totensor = transforms.ToTensor()
    img = Image.open(img).resize([inp_dim, inp_dim])
    return totensor(img).unsqueeze(0)

# dim = 608
# dim = 448
# dim = 416
# dim = 320
# dim = 256 
# dim = 224
dim = args.dim

device = 'cpu' # cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

model = DarkNet('./_model/yolov3.cfg', img_size=dim)
# model.load_state_dict(torch.load('./output/ckpt-00006'))
model.load_weights('./model/yolov3.weights')
model.eval()
model = model.to(device=device)

image = pre_image(args.path, dim)
data = image.to(dtype=torch.float32, device=device)

tic = time.time()
pred = model(data.to(device=device))
print('time: ', time.time()-tic)

show_bbox(Image.open(args.path).resize((dim, dim)), pred[pred[:, :, 0] > 0.3].cpu().data.numpy()[:, 1:5], xyxy=False, normalized=False)

pred = pred.cpu().data.numpy()[0]
result = NMS(pred, objectness_threshold=0.3, iou_threshold=0.8)

for i in result:
    show_bbox(Image.open(args.path).resize((dim, dim)), result[i][0])

# print('++++++++++++++++++++++++++++++++++++++++')
# import torch.onnx
# import caffe2.python.onnx.backend as backbend
# import onnx

# dummy_input = torch.zeros(1, 3, args.dim, args.dim)
# torch.onnx.export(model, dummy_input, 'test.onnx', verbose=True)
# model = onnx.load('test.onnx')
# onnx.checker.check_model(model)
# onnx.helper.printable_graph(model.graph) 

# rep = backbend.prepare(model, device='CPU')

# outs = rep.run(image.cpu().data.numpy().astype(np.float32))
# outs = outs[0][0]

# print(type(outs))
# print(outs[0].shape)

# show_bbox(Image.open(args.path).resize((dim, dim)), outs[outs[:, 0] > 0.3][:, 1:5], xyxy=False, normalized=False)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data 
from PIL import Image, ImageDraw
from torchvision.transforms import transforms
import time
import numpy as np
from model.network import BiSeNet
from data.ops_label_color import label_id_color
import glob
import random

device = torch.device('cuda:0')
# torch.backends.cudnn.benchmark = True

totensor = transforms.ToTensor()

model = BiSeNet(num_classes=5)
model.load_state_dict(torch.load('./output/state-ckpt-epoch-00200')['model'])
model.eval()
model.to(device=device)

paths = glob.glob('/home/wenyu/workspace/dataset/fisheye/val_anno/*.jpg')
random.shuffle(paths)
img = Image.open(paths[0]).resize((416, 416))

img.show()
img = totensor(img).unsqueeze(0)
img = img.to(device)

tic = time.time()
logits = model(img)
print('time: ', time.time() - tic)
print('logits: ', logits.shape)

prob = F.softmax(logits, dim=1)[0]
_, msk = prob.max(dim=0)
# print(msk)

msk = np.array(msk.cpu().data)
png = np.zeros(list(msk.shape)+[3], dtype=np.uint8)
# print('msk: ', msk.sum())

for k, v in label_id_color.items():
    # print(k, v)
    png[msk==k] = np.array(v).reshape(1, 1, 3)

Image.fromarray(png).show()
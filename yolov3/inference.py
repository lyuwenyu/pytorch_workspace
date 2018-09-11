import torch
import numpy as np
from PIL import Image, ImageDraw
from model.build_model import DarkNet
from torchvision import transforms
import time

def pre_image(img, inp_dim):
    totensor = transforms.ToTensor()
    img = Image.open(img).resize([inp_dim, inp_dim])
    return totensor(img).unsqueeze(0)

dim = 608
dim = 448
dim = 416
dim = 320
# dim = 256 
# dim = 224

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

img_path = './_model/dog-cycle-car.png'
image_pil = Image.open(img_path)
image_pil = image_pil.resize((dim, dim))
draw = ImageDraw.Draw(image_pil)

model = DarkNet('./_model/yolov3.cfg', img_size=dim)
# model.load_state_dict(torch.load('./model/yolov3.weights'))
model.load_weights('./model/yolov3.weights')
model.eval()
model = model.to(device=device)


image = pre_image(img_path, dim)
data = image.to(dtype=torch.float32, device=device)

tic = time.time()
pred = model(data.to(device=device))
print('time: ', time.time()-tic)

res = pred[pred[:, :, 4] > 0.99]

for p in res.cpu().data.numpy():
    draw.rectangle((p[0]-p[2]/2, p[1]-p[3]/2, p[0]+p[2]/2, p[1]+p[3]/2), outline='red')

image_pil.show()
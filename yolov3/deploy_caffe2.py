import torch.onnx
import caffe2.python.onnx.backend as backbend
import onnx
import torch
from torchvision import transforms

import numpy as np
from PIL import Image
from utils.ops_show_bbox import show_bbox

# dummy_input = torch.zeros(1, 3, 416, 416)
# torch.onnx.export(model, dummy_input, 'test.onnx', verbose=True)
def pre_image(img, inp_dim):
    totensor = transforms.ToTensor()
    img = Image.open(img).resize([inp_dim, inp_dim])
    return totensor(img).unsqueeze(0)

path = './_model/dog-cycle-car.png'
img_pil = Image.open(path).resize([416, 416])
img = pre_image(path, 416).data.numpy()

model = onnx.load('test.onnx')
onnx.checker.check_model(model)
# onnx.helper.printable_graph(model.graph) 
rep = backbend.prepare(model, device='CPU')

outs = rep.run(img)
outs = outs[0][0]

print(outs.shape)

show_bbox(img_pil, outs[outs[:, 0] > 0.9][:, 1:5], xyxy=False, normalized=False)


class Pytorch2caffe(object):
    
    def __init__(self, ):
        pass

    def export_pytorch(self, ):
        pass

    def inference_caffe(self, ):
        pass

import torch
import torch.nn as nn
from torch2trt import torch2trt
import torchvision.models as models

# model = models.resnet18(pretrained=True).eval().cuda()
model = nn.Conv2d(3, 10, 3, 2, 1)
x = torch.randn(1, 3, 224, 224).cuda()
model_trt = torch2trt(model, [x])


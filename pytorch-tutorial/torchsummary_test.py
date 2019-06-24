import torchvision.models as models
from torchsummary import summary

vgg = models.vgg13_bn()

summary(vgg, (3, 224 ,224))
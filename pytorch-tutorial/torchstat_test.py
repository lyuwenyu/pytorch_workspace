from torchstat import stat
from torchsummary import summary
import torchvision.models as models

model = models.resnet18()
stat(model, (3, 224, 224))

model = models.vgg13_bn()
summary(model, (3, 224 ,224))
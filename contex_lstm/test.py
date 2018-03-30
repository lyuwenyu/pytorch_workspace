import torch
import torchvision.models as models
import torch.nn as nn
from torch.autograd import Variable


mm = models.resnet50()
mm = nn.Sequential( *list(mm.children())[:-2] )

print(mm)

x = Variable( torch.randn(10, 3, 224, 224))
out = mm(x)

lstm = nn.LSTM(2048, 1024, 2, bidirectional=True)

out = torch.transpose(out, 0, 3).contiguous()
out = torch.transpose(out, 1, 3).contiguous()

print(out.size())

feas, _ = lstm( out.view(7, 10*7, 2048) )

feas = feas.view(7, 10, 7, -1)

feas = torch.transpose(feas, 1, 3).contiguous()
feas = torch.transpose(feas, 0, 3).contiguous()


print(mm(x).size())
print( feas.size() )
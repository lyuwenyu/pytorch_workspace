import torch
import torchvision.models as models
import torch.nn as nn
from torch.autograd import Variable


mm = models.resnet50()
mm = nn.Sequential( *list(mm.children())[:-2] )

print(mm)

x = Variable( torch.randn(10, 3, 224, 224))  # n c h w
out = mm(x)

lstm = nn.LSTM(2048, 1024, 2, bidirectional=True)

out = torch.transpose(out, 0, 3).contiguous() # w c h n
out = torch.transpose(out, 1, 3).contiguous() # w n h c

print(out.size())

feas, _ = lstm( out.view(7, 10*7, 2048) ) # w n*h c , seq_len batch feas_len

feas = feas.view(7, 10, 7, -1)

feas = torch.transpose(feas, 1, 3).contiguous() # w c h n
feas = torch.transpose(feas, 0, 3).contiguous() # n c h w


print(mm(x).size())
print( feas.size() )
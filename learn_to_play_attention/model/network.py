

import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class AttentionModel(nn.Module):
    
    def __init__(self, L=[2,3,4], num_classes=10, concat_feas=True):
        super(AttentionModel, self).__init__()

        self.L = L 
        self.concat_feas = concat_feas

        # 
        self.model = models.resnet101(pretrained=True)
        self.model.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.model.avgpool = nn.AdaptiveMaxPool2d(1)
        self.model.fc = nn.Linear(2048, 2048)
        
        self.linear_map = nn.ModuleList( [nn.Linear(256, 2048), nn.Linear(512, 2048), nn.Linear(1024, 2048), nn.Linear(2048, 2048) ] )


        if concat_feas:
            self.fc_cls = nn.Linear(len(L)*2048, num_classes)
        else:
            self.fc_cls = nn.ModuleList( [nn.Linear(2048, num_classes)]*len(L))


    def forward(self, x):
        
        feas = [] # local features
        g = None  # global features
        
        for i, (n, m) in enumerate(self.model.named_children()):

            if n in [ 'layer{}'.format(l) for l in self.L ]:

                x = m(x)
                n, c, h, w = x.size()
                fea = x.permute(0, 2, 3, 1).contiguous().view(n, h*w, c)
                fea = self.linear_map[i-4](fea)
        
                feas += [fea]

            elif n == 'fc':

                n, c, _, _ = x.size()
                g = m(x.view(n, c))

            # elif n == 'avgpool':

            #     n, c, _, _ = x.size()
            #     g = m(x).view(n, c)
            
            else:

                x = m(x)


        a_s = [ F.softmax(a, dim=1) for a in self._c(feas, g) ]

        gas = [ a.view(*a.size(), 1)*f for (a, f) in zip(a_s, feas)]
        gas = [ torch.sum(g_, dim=1) for g_ in gas]

        if self.concat_feas:

            feas = torch.cat(gas, dim=1)
            logits = self.fc_cls(feas)
        
        else:

            logits = [ m_(x_) for x_, m_ in zip(feas, self.fc_cls)] 
            
            
        return logits


    def _c(self, vs1, v2, mode='dot'):
        

        if not isinstance(vs1, list):

            raise RuntimeError

        outs = []

        if mode == 'dot':

            for v1 in vs1:

                b, n, c = v1.size()

                v_tmp = v2.view(b, 1, c) + torch.zeros_like(v1)

                out = torch.bmm(v_tmp.view(b*n, 1, c), v1.view(b*n, c, 1)).view(b, n)

                outs += [out]

                print(out.size())

        else:
            pass


        return outs


if __name__ == '__main__':

    model = AttentionModel()

    x = Variable(torch.randn(5, 3, 224,224))

    if torch.cuda.is_available():
        model = model.cuda()
        x = x.cuda()


    out = model(x)

    print(out.size())

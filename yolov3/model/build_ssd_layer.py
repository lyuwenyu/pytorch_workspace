import torch
import torch.nn as nn
from utils.ops_priors_bbox import PriorBox, match

class SSDLayer(nn.Module):
    def __init__(self, ):
        super(SSDLayer, self).__init__()

        self.priorbox = PriorBox()()
    
        self.smoothl1loss = nn.SmoothL1Loss()
        self.crossentropy = nn.CrossEntropyLoss()
    
    def forward(self, features, target=None):
        '''
        p = [features, features, features]
        '''
        p = torch.cat([f.view(-1, f.shape[1] * f.shape[2], -1) for f in features], dim=1)

        if target is None:
            pass

        else:
            
            loss = 0
            for i in range(p.shape[0]):
                locs, labs = match(target[i], target[i], self.priorbox, threshold=0.5)

                loss += 0
                pass
            
            return loss

        

if __name__ == '__main__':

    ssdlayer = SSDLayer()

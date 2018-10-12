import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.ops_priors_bbox import PriorBox, match

class SSDLayer(nn.Module):
    def __init__(self, ):
        super(SSDLayer, self).__init__()

        self.priorbox = PriorBox()()
        self.num_priors = self.priorbox.shape[0]
        self.num_classes = 10
        self.num_attr = self.num_classes + 4
        self.negpos_ratio = 3

        self.smoothl1loss = nn.SmoothL1Loss(size_average=False)
        self.crossentropy = nn.CrossEntropyLoss(size_average=False)
    
    def forward(self, features, targets=None):
        '''
        features: [features, features, features]: n num_class+4 h w 
        target: [lab, x1 y1 x2 y2]: [n_obj * 5] * batch
        '''

        p = torch.cat([f.view(f.shape[0], self.num_attr, -1).permute(0, 2, 1).contiguous() for f in features], dim=1)

        loc_p = p[:, :, :4]
        conf_p = p[:, :, 4:]

        if targets is None:
            pass

        else:

            loc_t = torch.zeros(p.shape[0], self.num_priors, 4).to(dtype=p.dtype, device=p.device)
            conf_t = torch.zeros(p.shape[0], self.num_priors).to(dtype=p.dtype, device=p.device)
            for i in range(p.shape[0]):
                locs, labs = match(gts=targets[i][1:], labels=targets[i][0], priors=self.priorbox, threshold=0.5)
                loc_t[i] = locs
                conf_t[i] = labs
            
            pos_idx = conf_t > 0
            num_pos = pos_idx.sum()

            loss_loc = self.smoothl1loss(loc_t[pos_idx], loc_p[pos_idx])

            # hard neg mining
            loss_c, _ = torch.logsumexp(conf_p, dim=-1) - conf_p.gather(2, conf_t.unsqueeze(dim=-1).suqeeze())
            loss_c[pos_idx] = 0
            _, loss_idx = loss_c.sort(dim=1, descending=True)
            _, idx_rank = loss_idx.sort(dim=1)

            num_neg = torch.clamp(self.negpos_ratio*num_pos, max=self.num_priors-num_pos)
            neg_idx = idx_rank < num_neg
            
            loss_c = self.crossentropy(conf_p[pos_idx + neg_idx], conf_t[pos_idx + neg_idx])
            
            loss = (loss_loc + loss_c) / num_pos

            return loss
        

if __name__ == '__main__':

    ssdlayer = SSDLayer()

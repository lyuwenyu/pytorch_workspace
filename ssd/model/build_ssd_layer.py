import torch
import torch.nn as nn
import torch.nn.functional as F

import os, sys
from .ops_priors_bbox import PriorBox, match
from .config import cfg

class SSDLayerLoss(nn.Module):
    def __init__(self, ):
        super(SSDLayerLoss, self).__init__()

        # num_priors = self.priorbox.shape[0]

        self.smoothl1loss = nn.SmoothL1Loss()
        self.crossentropy = nn.CrossEntropyLoss()
    
    def forward(self, loc_p, conf_p, priorbox, targets):
        '''
        loc_p: n priors 4
        conf_p: n priors num_classes
        priorbox: priors 4
        target: [lab, x1 y1 x2 y2]: [n_obj * 5] * n
        '''
        num_priors = priorbox.shape[0]
        negpos_ratio = cfg['negpos_ratio']
        n = loc_p.shape[0]
        dtype = loc_p.dtype
        device = loc_p.device
        num_classes = conf_p.shape[-1]

        loc_t = torch.zeros(n, num_priors, 4).to(dtype=dtype, device=device)
        conf_t = torch.zeros(n, num_priors).to(dtype=dtype, device=device)
        # print('loc_t: ', loc_t.shape)
        # print('conf_t', conf_t.shape)

        for i in range(n):
            
            target = targets[i]
            target = target[target[:, 0] > 0][:, 1:]

            locs, labs = match(gts=target[:, 1:], labels=target[:, 0], priors=priorbox, threshold=0.5)
            loc_t[i] = locs
            conf_t[i] = labs

            # print(locs.shape, labs.shape)
        
        pos_idx = conf_t > 0
        num_pos = pos_idx.sum()

        loss_loc = self.smoothl1loss(loc_p[pos_idx], loc_t[pos_idx]) # order should not change
        # print('loss_loc: ', loss_loc.item())

        # hard negative mining
        # loss_c = torch.logsumexp(conf_p, dim=-1) - torch.gather(conf_p, dim=-1, index=conf_t.unsqueeze(dim=-1).long()).squeeze()
        # loss_c[pos_idx] = 0
        # _, loss_idx = loss_c.sort(dim=1, descending=True)
        # _, idx_rank = loss_idx.sort(dim=1)
        # num_neg = torch.clamp(negpos_ratio * num_pos, max=num_priors - num_pos)
        # neg_idx = idx_rank < num_neg
        # loss_c = self.crossentropy(conf_p[pos_idx + neg_idx], conf_t[pos_idx + neg_idx].long())
        
        # print(conf_p.shape)
        # print(conf_t.shape)

        loss_c = self.crossentropy(conf_p.view(-1, num_classes), conf_t.long().view(-1))

        # weight = torch.ones(num_classes, dtype=dtype, device=device)
        # weight[1:] = ((n * num_priors - num_pos) * (num_classes-1)) / num_pos 
        # loss_c = F.cross_entropy(conf_p.view(-1, 21), conf_t.view(-1).long(), weight=weight)

        # print(loss_loc)
        # print(loss_c)

        loss = loss_loc + loss_c #  / num_pos.float()

        return loss
    

if __name__ == '__main__':

    # ssdlayer = SSDLayerLoss(10)
    # print(ssdlayer)
    pass
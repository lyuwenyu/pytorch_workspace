import torch
import torch.nn as nn
import torch.nn.functional as F

import os, sys
from ops_priors_bbox import PriorBox, match

class SSDLayer(nn.Module):
    def __init__(self, attr_num=0):
        super(SSDLayer, self).__init__()

        self.attr_num = attr_num
        self.priorbox = PriorBox()(VISUALIZATION=False)
        self.num_priors = self.priorbox.shape[0]

        self.negpos_ratio = 3

        self.smoothl1loss = nn.SmoothL1Loss()
        self.crossentropy = nn.CrossEntropyLoss()
    
    def forward(self, p, targets=None):
        '''
        features: batch n 4 + num_class + 1
        target: [lab, x1 y1 x2 y2]: [n_obj * 5] * batch
        '''
        # outs = [out.view(out.shape[0], -1, self.attr_num, out.shape[2], out.shape[3]).permute(0, 1, 3, 4, 2).contiguous() for out in features]
        # outs = [out.view(out.shape[0], -1, self.attr_num).contiguous() for out in outs]
        # p = torch.cat(outs, dim=1)
        
        self.priorbox =self.priorbox.to(device=p.device)
        
        loc_p = p[:, :, :4]
        conf_p = p[:, :, 4:]

        if targets is None:
            
            out = torch.zeros_like(p)
            out[:, :, :2] = (loc_p[:, :, :2] + 1) * self.priorbox.unsqueeze(0)[:, :, 2:]
            out[:, :, 2:4] = torch.exp(loc_p[:, :, 2:]) * self.priorbox.unsqueeze(0)[:, :, 2:]
            out[:, :, 4:] = conf_p
            
            return out

        else:

            loc_t = torch.zeros(p.shape[0], self.num_priors, 4).to(dtype=p.dtype, device=p.device)
            conf_t = torch.zeros(p.shape[0], self.num_priors).to(dtype=p.dtype, device=p.device)
            # print('loc_t: ', loc_t.shape)
            # print('conf_t', conf_t.shape)

            for i in range(p.shape[0]):
                
                target = targets[i]
                target = target[target[:, 0] > 0][:, 1:]

                locs, labs = match(gts=target[:, 1:], labels=target[:, 0], priors=self.priorbox, threshold=0.5)
                loc_t[i] = locs
                conf_t[i] = labs

                # print(locs.shape, labs.shape)
            
            pos_idx = conf_t > 0
            num_pos = pos_idx.sum()
            # print(num_pos)
            # print('pos_idx: ', pos_idx.shape)
            # print('num_pos: ', num_pos)
            # print(loc_t.shape)
            # print(loc_p.shape)

            loss_loc = F.smooth_l1_loss(loc_p[pos_idx], loc_t[pos_idx]) # order should not change

            # hard neg mining
            loss_c = torch.logsumexp(conf_p, dim=-1) - conf_p.gather(2, conf_t.unsqueeze(dim=-1).long()).squeeze()
            # print(loss_c.shape)
            # print(pos_idx.dtype)

            loss_c[pos_idx] = 0
            _, loss_idx = loss_c.sort(dim=1, descending=True)
            _, idx_rank = loss_idx.sort(dim=1)

            num_neg = torch.clamp(self.negpos_ratio * num_pos, max=self.num_priors - num_pos)
            neg_idx = idx_rank < num_neg

            # print(conf_t[pos_idx + neg_idx].shape)
            # print(conf_p[pos_idx + neg_idx].shape)

            # loss_c = F.cross_entropy(conf_p[pos_idx + neg_idx], conf_t[pos_idx + neg_idx].long())
            loss_c = self.crossentropy(conf_p[pos_idx + neg_idx], conf_t[pos_idx + neg_idx].long())
            
            # print(loss_loc)
            # print(loss_c)

            loss = loss_loc + loss_c #  / num_pos.float()

            return loss
        

if __name__ == '__main__':

    ssdlayer = SSDLayer(10)
    print(ssdlayer)
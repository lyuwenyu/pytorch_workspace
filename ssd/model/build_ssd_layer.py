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

        # self.smoothl1loss = nn.SmoothL1Loss(reduction='sum')
        # self.crossentropy = nn.CrossEntropyLoss(reduction='sum')
        pass
    
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
        # num_classes = conf_p.shape[-1]

        loc_t = torch.zeros(n, num_priors, 4).to(dtype=dtype, device=device)
        conf_t = torch.zeros(n, num_priors).to(dtype=dtype, device=device)
        # print('loc_t: ', loc_t.shape)
        # print('conf_t', conf_t.shape)

        for i in range(n):
            
            _target = targets[i]
            _target = _target[_target[:, 0] > 0][:, 1:]

            locs, labs = match(gts=_target[:, 1:], labels=_target[:, 0], priors=priorbox, threshold=0.5)
            loc_t[i] = locs
            conf_t[i] = labs

            # print(locs.shape, labs.shape)
        
        pos = conf_t > 0
        num_pos = pos.sum(dim=-1, keepdim=True)
        
        loss_loc = F.smooth_l1_loss(loc_p[pos], loc_t[pos], size_average=False) # order should not change

        # hard negative mining
        if True:

            _loss_c = torch.logsumexp(conf_p.data, dim=-1)
            _loss_c -= conf_p.data.gather(dim=-1, index=conf_t.data.view(n, num_priors, 1).expand_as(conf_p.data).long())[:, :, 0]

            _loss_c[pos] = 0

            _, loss_idx = _loss_c.sort(dim=-1, descending=True)
            _, idx_rank = loss_idx.sort(dim=-1)
            num_neg = torch.clamp(negpos_ratio * num_pos, max=num_priors - 1)
            neg = idx_rank < num_neg
            # _, neg = loss_c.topk(k=num_neg, dim=-1)

            loss_c = F.cross_entropy(conf_p[(neg + pos) > 0], conf_t[(pos + neg) > 0].long(), size_average=False)
            
        ## focal loss
        # else:
        #     pos_t = conf_t > -1
        #     mask = pos_t.unsqueeze(2).expand_as(conf_p)
        #     masked_pred_conf = conf_p[mask].view(-1, num_classes)
        #     loss_c = self.focal_loss(masked_pred_conf, conf_t[pos_t].long(), num_classes)

        loss = (loss_loc + loss_c) / num_pos.data.sum().float()

        return loss
    
    def focal_loss(self, x, y, num_classes):
        '''
        x: N D
        y: N
        '''
        alpha = 0.25
        gamma = 2

        t = self._one_hot_embeding(y, num_classes)

        logits = F.softmax(x)
        logits = logits.clamp(1e-7, 1.-1e-7)
        conf_loss_tmp = -1 * t.float() * torch.log(logits)
        conf_loss_tmp = alpha * conf_loss_tmp * (1 - logits) ** gamma
        conf_loss_sum = conf_loss_tmp.sum()

        return conf_loss_sum

    def _one_hot_embeding(self, labels, num_classes):
        '''
        '''
        y = torch.eye(num_classes).to(device=labels.device)
        return y[labels]

    def log_sum_exp(self, x):
        ''''''
        x_max = x.data.max()
        return torch.log(torch.sum(torch.exp(x - x_max), dim=1, keepdim=True)) + x_max


if __name__ == '__main__':

    # ssdlayer = SSDLayerLoss(10)
    # print(ssdlayer)
    pass
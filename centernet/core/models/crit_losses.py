import torch
import torch.nn as nn



def giou_loss(bboxa, bboxb):
    '''
    bboxa: n x 2
    bboxb: n x 2
    '''
    bboxa.clamp_(min=0); bboxb.clamp_(min=0)
    bboxa = bboxa.view(-1, 2)
    bboxb = bboxb.view(-1, 2)
    areaa = bboxa.prod(dim=-1)
    areab = bboxb.prod(dim=-1)
    ovlap = torch.min(bboxa[:, 0], bboxb[:, 0]) * torch.min(bboxa[:, 1], bboxb[:, 1])
    areac = torch.max(bboxa[:, 0], bboxb[:, 0]) * torch.max(bboxa[:, 1], bboxb[:, 1])
    union = areaa + areab - ovlap
    giou = ovlap / union - (1 - union / areac)
    loss = (1 - giou).sum() / giou.size(0)
    
    return loss
    
    
def binary_focal_loss_with_logits(p, gt):
    '''
    '''
    p = p.sigmoid()
    pos_loss = torch.log(p + 1e-15) * (1 - p).pow(2) * (gt==1).float()
    neg_loss = (1 - gt).pow(4) * torch.log(1 - p + 1e-15) * p.pow(2) * (gt!=1).float()
    loss = -(pos_loss.sum() + neg_loss.sum()) / (gt==1).sum().float()

    return loss

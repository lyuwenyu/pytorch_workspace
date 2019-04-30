import torch


def binary_focal_loss_with_logits(p, gt):
    '''
    '''
    p = p.sigmoid()
    pos_loss = torch.log(p + 1e-15) * (1 - p).pow(2) * (gt==1).float()
    neg_loss = (1 - gt).pow(4) * torch.log(1 - p + 1e-15) * p.pow(2) * (gt!=1).float()
    loss = -(pos_loss.sum() + neg_loss.sum()) / (gt==1).sum().float()

    return loss

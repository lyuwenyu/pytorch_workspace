import torch


def focal_loss_with_logits(p, t, alpha=1., gamma=2., epsilon=1e-8, pos_weight=1., reduction='mean'):
    '''
    '''
    p_sigmoid = p.sigmoid()
    t_smooth = (1 - epsilon) * t + epsilon / 2
    pt = (1 - p_sigmoid) * t + p_sigmoid * (1 - t)
    # w = (alpha * t + (1 - alpha) * (1 - t)) * pt.pow(gamma)
    w = (pos_weight * t + (1 - t)) * pt.pow(gamma)
    loss = w * (-t_smooth * torch.log(p_sigmoid + 1e-10) - (1 - t_smooth) * torch.log(1 - p_sigmoid + 1e-10))

    if reduction == 'mean':
        return loss.sum() / t.sum().float()



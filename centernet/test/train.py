import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data

from core.datasets import dataset_factory
from core.models import crit_losses as crit
from core.models import pose_dla_dcn_origin as pose_dla_dcn 

from config import test_config as cfg

import time

print(cfg.dataset)

device = torch.device('cuda:0')


def train(model, dataloader, optimizer, scheduler, epoch=0):
    model.train()
    scheduler.step()

    for i, blob in enumerate(dataloader):
    
        blob = {k: v.to(device) for k, v in blob.items()}
        mask = blob['mask']

        result =  model(blob['image'])
        
        losses = {}
        for k in result:
            p = result[k].permute(0, 2, 3, 1)
            if k == 'hm':
                # losses[k] = F.binary_cross_entropy_with_logits(p, blob[k])
                losses[k] = crit.binary_focal_loss_with_logits(p, blob[k])
            elif k == 'wh':
                losses[k] = 0.2 * F.l1_loss(p[mask], blob[k][mask])
                # losses[k] = crit.giou_loss(p[mask], blob[k][mask]) 
            elif k == 'off':
                p = p.sigmoid()
                losses[k] = F.l1_loss(p[mask], blob[k][mask])

        tloss = sum(losses.values())
        
        optimizer.zero_grad()
        tloss.backward()
        optimizer.step()
        print({k: format(v.item(),'.4f') for k, v in losses.items()})

    if epoch % 5 == 0:
        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        torch.save(state, './tmp/state-ckpt-epoch-{:0>5}'.format(epoch))


if __name__ == '__main__':

    dataset = dataset_factory.get_dataset(cfg.dataset)
    dataloader = data.DataLoader(dataset, batch_size=12, num_workers=6, shuffle=True)
    
    mm = pose_dla_dcn.get_pose_net('34', cfg.network['heads'], 1)
    mm = mm.to(device)

    optimizer = optim.SGD(mm.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80, 100], gamma=0.1)

    for e in range(scheduler.last_epoch + 1, 120):
        train(mm, dataloader, optimizer, scheduler, e)


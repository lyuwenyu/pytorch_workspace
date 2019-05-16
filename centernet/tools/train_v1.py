import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.utils as vutils
# from torchstat import stat
# from torchsummary import summary
import onnx
import time
import random
import glob
import numpy as np
import apex
import torch.distributed as tdist
from tensorboardX import SummaryWriter

import torch.backends.cudnn as cudnn

from data.dataset import Dataset
from utils.ops_weight_init import resume
from utils.ops_show_bbox import show_tensor_bbox

import config

# torch.manual_seed(0)
# torch.cuda.manual_seed_all(0)
# python -m torch.distributed.launch --nproc_per_node 2 train_v1.py --dcn --warmup --multi-scale --distributed

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--device-ids', type=list, default=[0, 1])
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--num-classes', type=int, default=config.num_classes)
parser.add_argument('--resume-path', type=str, default='')
parser.add_argument('--loss-step', type=int, default=10)
parser.add_argument('--save-step', type=int, default=10)
parser.add_argument('--img-dims', type=list, default=list(range(320, 640+32, 32)))
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--accumulation-steps', type=int, default=1)
parser.add_argument('--num-workers', type=int, default=8)
parser.add_argument('--epoches', type=int, default=151)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=0.0001)
parser.add_argument('--milestones', type=list, default=[50, 80, 100])
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--mixup', action='store_true')
parser.add_argument('--warmup', action='store_true')
parser.add_argument('--multi-scale', action='store_true')
parser.add_argument('--dynamic-anchor', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--amp', action='store_true')

parser.add_argument('--project-name', type=str, default='')
parser.add_argument('--dcn', action='store_true')

parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--sync-bn', action='store_true')
parser.add_argument('--distributed', action='store_true')

args = parser.parse_args()

writer = SummaryWriter('./logs')
global_step = -1
    
device = 'cuda:{}'.format(args.local_rank) if torch.cuda.is_available() and not args.cpu else 'cpu'
device = torch.device(device)
print(device)

if args.amp:
    from apex import amp

if args.distributed:
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    args.world_size = torch.distributed.get_world_size()
    cudnn.benckmark = True
    torch.cuda.set_device(args.local_rank)
    print('world_size: ', args.world_size)
    print('local_rank: ', args.local_rank, torch.distributed.get_rank())

print(args)
print('--------init done...----------')


def train(model, dataloader, optimizer, scheduler, epoch=0):
    '''train'''
    model.train()
    model.zero_grad()
    scheduler.step()

    step_loss = []
    step_time = []
    N = len(dataloader)

    for k, (images, targets) in enumerate(dataloader):
        global global_step
        global_step += 1
        images = images.to(device)
        targets = targets.to(device)
        
        # TODO warmup
        if args.warmup and epoch == 0:
            warmup_step = 250
            if k == 0:
                for g in optimizer.param_groups:
                    g['lr'] = args.lr * 0.1 ** (N // warmup_step)
            elif k % warmup_step == 0:
                for g in optimizer.param_groups:
                    g['lr'] *= 10
                    g['lr'] = args.lr if g['lr'] > args.lr else g['lr']
        
        # TODO multi-scale
        scale = 1.
        if args.multi_scale and epoch > 5 and epoch < args.epoches - 5:
            dim = random.choice(args.img_dims)
            if args.dynamic_anchor:
                scale = dim / config.base_dim
            images = F.interpolate(images, size=(dim, dim))
            
        # TODO mixup
        if args.mixup and epoch > 10:
            lam = np.random.beta(1.2, 1.2)
            index = torch.randperm(images.size(0)).to(device)
            images = lam * images + (1 - lam) * images[index]
            
        # TODO debug
        if args.debug:
            if args.mixup:
                targets = torch.cat([targets, targets[index]], dim=1)
            for im, tar in zip(images, targets):
                show_tensor_bbox(im, tar)
        
        # TODO multi-gpu
        tic = time.time()
        if args.distributed:
            feats = model(images)
        elif not args.cpu and len(args.device_ids) > 1:
            feats = nn.parallel.data_parallel(module=model, inputs=images, device_ids=args.device_ids, dim=0)
        else:
            feats = model(images)
        
        # TODO compute loss
        losses = []
        for i, p in enumerate(feats):
            scaled_anchors = scale * torch.tensor(config.anchors[i]).to(dtype=torch.float, device=p.device) / config.strides[i]
            if args.mixup and epoch > 10:
                losses1 = yololayer(p, targets, scaled_anchors, args.num_classes, config.strides[i])
                losses2 = yololayer(p, targets[index], scaled_anchors, args.num_classes, config.strides[i])
                losses += [tuple(lam * l1 + (1-lam) * l2 for l1, l2 in zip(losses1, losses2)), ]
            else:
                losses += [yololayer(p, targets, scaled_anchors, args.num_classes, config.strides[i])]

        if args.local_rank == 0:
            writer.add_scalar('train/lr', optimizer.param_groups[-1]['lr'], global_step)
            logs = [x.item()+y.item()+z.item() for x, y, z in zip(*losses)]
            names = ['loss', 'lx', 'ly', 'lw', 'lh', 'lconf', 'lcls', 'lgiou', 'numobj', 'numneg', 'numign']
            scalars = {k: v for k, v in zip(names, logs)}
            print('{:0>6}:'.format(global_step), [format(log, '.4f') for log in logs[:-3]])
        
            for kk, vv in scalars.items():
                writer.add_scalar('/loss/{}'.format(kk), vv, global_step)
        
        loss = sum(lss[0] for lss in losses) / args.accumulation_steps
        
        if args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()   
        else:
            loss.backward()
        
        # TODO accumulation gradient
        if (k + 1) % args.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        step_loss += [loss.item(), ]
        step_time += [time.time() - tic]

        if (k + 1) % args.loss_step == 0:
            step_loss = np.array(step_loss).sum() / args.loss_step
            lin = 'epoch/iter: {:0>3}/{:0>3}, lr: {}, time: {:.4f}s, losses: {:.4f}'.format(epoch, k + 1,
                                                                                format(optimizer.param_groups[-1]['lr'],'.8f'),
                                                                                np.array(step_time).mean(),
                                                                                step_loss)
            step_loss, step_time = [], []
            print(lin)

            _images = [show_tensor_bbox(im, tar) for im, tar in zip(images, targets)]
            _images = vutils.make_grid(_images, nrow=2, padding=5, normalize=True, scale_each=True)
            writer.add_image('vis/imagesWITHtargets', _images, global_step)
        
    # TODO save state
    if epoch % args.save_step == 0 and args.local_rank == 0:
        if args.distributed:
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
            
        state = {
            'epoch': epoch,
            'model': model_state,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        
        torch.save(state, './output/{}-state-ckpt-epoch-{:0>5}'.format(args.project_name, epoch))

        # for n, p in model.named_parameters():
        #     writer.add_histogram(n, p.clone().cpu().data.numpy(), global_step)

        _images = [show_tensor_bbox(im, tar) for im, tar in zip(images, targets)]
        _images = vutils.make_grid(_images, nrow=4, padding=5, normalize=True, scale_each=True)
        writer.add_image('vis/imagesWITHtargets', _images, global_step)

    print(''.join(['-'] * len(lin)))

    
def export_onnx(model):
    '''
    '''
    model.eval()
    model.to(device)
    dummy = torch.ones(1, 3, 256, 320).to(device)
    torch.onnx.export(model, dummy, 'yolov3.onnx', export_params=True, verbose=False)
    onnx.checker.check_model(onnx.load('yolov3.onnx'))


def reduce_tensor(tensor, local_rank):
    '''
    '''
    x = tensor.clone()
    # tdist.all_reduce(x, op=tdist.reduce_op.SUM)
    tdist.reduce(x, local_rank, op=tdist.reduce_op.SUM)
    return x / args.world_size


def collate_fn(batch):
    '''
    '''
    pass



if __name__ == '__main__':

    datasets = Dataset('', size=(416, 416)) # W H
    sampler = None
    if args.distributed:
        sampler = data.distributed.DistributedSampler(datasets)

    dataloader = data.DataLoader(datasets, 
                                  batch_size=args.batch_size,
                                  shuffle=(sampler is None),
                                  num_workers=args.num_workers,
                                  sampler=sampler)
    if args.dcn:
        model = yolov3_darknet_dcn(num_classes=args.num_classes, blocks=(1, 2, 8, 8, 4), heads=(3, 3, 3), use_decov=True, phase='train')
        offset_params = [m for n, m in model.named_parameters() if 'offset' in n]
        normal_params = [m for n, m in model.named_parameters() if 'offset' not in n]
        optimizer = optim.SGD([{'params': offset_params, 'lr': args.lr * 0.1}, {'params': normal_params}], \
                              lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        model = yolov3_darknet(num_classes=args.num_classes, blocks=(1, 2, 8, 8, 4), heads=(3, 3, 3), use_decov=True, phase='train')
        model = model.cuda()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.sync_bn:
        model = apex.parallel.convert_syncbn_model(model)
        print('using apex sync bn...')
    
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoches)
    
    if args.resume_path:
        resume(model, optimizer, scheduler, state_path=args.resume_path,) 
        print('resume state from {}..'.format(args.resume_path))  
        
    if args.amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        print('mixed precision for training...')
        
    if args.distributed:
        print('distributed...')
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
        # model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)
        # print(model)
        
    print('start training...')

    e = scheduler.last_epoch + 1
    for e in range(e, args.epoches):
        if args.distributed: 
            sampler.set_epoch(e)
        train(model, dataloader, optimizer, scheduler, e)
    
    if args.local_rank == 0:
        torch.save(model.state_dict(), './output/freebies/{}-state-ckpt-epoch-final'.format(args.project_name))
        writer.export_scalars_to_json('./logs/{}-all-scalars.json'.format(args.project_name))
        writer.close()

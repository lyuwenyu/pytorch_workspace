"""
python -m torch.distributed.launch --nproc_per_node=2 distributed.py
"""
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.utils.data
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import ReduceOp

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=24)
args = parser.parse_args()
args.ngpus_per_node = torch.cuda.device_count()
print(args)

def init_dist():
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(args.local_rank)

def tensor_reduce(t, dst=0):
    _t = t.clone()
    dist.all_reduce(_t, op=ReduceOp.SUM)
    return _t

def tensor_gather(t, dst=0):
    _t = [torch.zeros_like(t) for _ in range(2)]
    dist.all_gather(_t, t)
    return _t

def dataset_loader():
    '''
    '''
    dataset = torch.utils.data.Dataset()
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size//ngpus_per_node, num_workers=8, sampler=sampler)
    return loader


if __name__ == '__main__':

    init_dist()
    
    m = nn.Conv2d(3, 10, 3, 1, 1)
    m.cuda()

    m = DDP(m, device_ids=[args.local_rank], output_device=args.local_rank)
    print(m.module)

    im = torch.randn(10, 3, 100, 100).cuda()
    data = m(im)

    t = tensor_gather(data)

    if args.local_rank == 0:
        for tt in t:
            print(tt.sum(), tt.shape)




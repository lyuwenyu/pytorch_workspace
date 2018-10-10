import numpy as np 
import torch

def bbox_overlap(box_a, box_b, x1y1x2y2=True):
    '''
    A X 4 []
    '''
    A = box_a.shape[0]
    B = box_b.shape[0]

    # functional euqal to repeat, but without alloc new mem.
    min_points = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                            box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    max_points = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                            box_b[:, 2:].unsqueeze(0).expand(A, B, 2))

    inter = torch.clamp(max_points-min_points, min=0)

    return inter.prod(dim=-1)


def jaccard(boxa, boxb):
    '''
    A X 4  e.g. [gts, 4]
    B X 4  e.g. [priors, 4]
    '''

    inter = bbox_overlap(boxa, boxb)
    area_a = ((boxa[:, 2] - boxa[:, 0]) * (boxa[:, 3] - boxa[:, 1])).unsqueeze(1).expand_as(inter)
    area_b = ((boxb[:, 2] - boxb[:, 0]) * (boxb[:, 3] - boxb[:, 1])).unsqueeze(0).expand_as(inter)

    union = area_a + area_b - inter
    return inter / union



if __name__ == '__main__':

    box1 = torch.rand(3, 4)
    box2 = torch.rand(10, 4)

    inters = bbox_overlap(box1, box2)
    print(inters.shape)

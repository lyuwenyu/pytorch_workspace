import numpy as np 
import torch
import math
import time

import torch

class PriorBox(object):
    def __init__(self, ):
        
        self.img_dim = 320
        self.strides = [8, 16, 32]
        self.grids = [int(self.img_dim / s) for s in self.strides]

        # anchor size
        self.base_sizes = [60, 120, 180]
        self.area_rarios = [[2, ], [2, ], [2, ]]
        self.aspect_rarios = [[2, ], [2, 3], [2, 3]]

    def __call__(self, ):
        '''
        generate prior bbox, [cx, cy, w, h] and normalized
        return: torch.Tensor
        '''
        anchors = []
        for i, grid in enumerate(self.grids):
            
            # center_x center_y
            yy, xx = np.meshgrid(range(grid), range(grid))
            # yy, xx = torch.meshgrid((torch.arange(grid), torch.arange(grid)))
            cx = (xx + 0.5) / grid
            cy = (yy + 0.5) / grid
        
            # width height
            whs = []
            base = self.base_sizes[i] / self.img_dim
            whs += [[base, base]]

            for ar in self.aspect_rarios[i]:
                whs += [[base * math.sqrt(ar), base / math.sqrt(ar)]]
                whs += [[base / math.sqrt(ar), base * math.sqrt(ar)]]
            
            for ar in self.area_rarios[i]:
                whs += [[base * math.sqrt(ar), base * math.sqrt(ar)]]
                whs += [[base / math.sqrt(ar), base / math.sqrt(ar)]]

            whs = np.array(whs)
            # whs = torch.from_numpy(np.array(whs))
            # print(len(whs))

            # n = xx.shape[0] * xx.shape[1]
            priors = np.zeros((np.prod(cx.shape), len(whs), 4))
            # print(priors.shape)
            # priors = torch.zeros((n, len(whs), 4))
            priors[:, :, 0] = cx.reshape(-1, 1)
            priors[:, :, 1] = cy.reshape(-1, 1)
            priors[:, :, 2:] = whs

            anchors += [priors.reshape(-1, 4)]
            # print(np.prod(cx.shape) * len(whs))

        anchors = np.concatenate(anchors, axis=0)
        # anchors = torch.cat(anchors, dim=0)
        # print(anchors.shape)
        anchors = torch.from_numpy(anchors).to(dtype=torch.float)
        # anchors.clamp_(max=1., min=0.)
        # print('anchors: ', anchors.shape)
        return anchors


def bbox_overlap(box_a, box_b, x1y1x2y2=True):
    '''
    A X 4  e.g. [gts, 4]
    B X 4  e.g. [priors, 4]
    return A X B
    '''
    A = box_a.shape[0]
    B = box_b.shape[0]

    # functional euqal to repeat, but without alloc new mem.
    min_points = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                            box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    max_points = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                            box_b[:, 2:].unsqueeze(0).expand(A, B, 2))

    inter = torch.clamp(max_points - min_points, min=0)

    return inter.prod(dim=-1)


def jaccard(boxa, boxb):
    '''
    A X 4  e.g. [gts, 4]
    B X 4  e.g. [priors, 4]
    return A X B
    '''
    inter = bbox_overlap(boxa, boxb)
    area_a = ((boxa[:, 2] - boxa[:, 0]) * (boxa[:, 3] - boxa[:, 1])).unsqueeze(1).expand_as(inter)
    area_b = ((boxb[:, 2] - boxb[:, 0]) * (boxb[:, 3] - boxb[:, 1])).unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter

    return inter / union


def xywh2xyxy(bbox):
    '''
    '''
    new_bbox = torch.zeros_like(bbox)
    new_bbox[:, 0] = bbox[:, 0] - bbox[:, 2] / 2
    new_bbox[:, 1] = bbox[:, 1] - bbox[:, 3] / 2
    new_bbox[:, 2] = bbox[:, 0] + bbox[:, 2] / 2
    new_bbox[:, 3] = bbox[:, 1] + bbox[:, 3] / 2

    return new_bbox


def match(gts, labels, priors, threshold=0.5):
    '''
    '''
    assert gts.shape[0] == labels.shape[0], ''

    overlaps = jaccard(gts, xywh2xyxy(priors))

    # print(overlaps.shape)

    _, best_prior_idx = overlaps.max(1) # best prior for each gt
    best_gt_overlap, best_gt_idx = overlaps.max(0) # best gt for each prior

    best_gt_overlap.index_fill_(0, best_prior_idx, 1) # for threshold 
    for j in range(best_prior_idx.size(0)): # 
        best_gt_idx[best_prior_idx[j]] = j

    matches = gts[best_gt_idx]
    print(matches.shape)

    # print(len(best_gt_overlap<=0.5))
    # print(torch.sum(best_gt_overlap<=0.5)) #HERE, sum difference to torch.sum
    # print(sum(best_gt_overlap<=0.5))

    # labels target
    clss = labels[best_gt_idx] + 1
    clss[best_gt_overlap < threshold] = 0 # bg
    # print((clss==0).sum())

    # center offet target
    t_cxcy = (matches[:, :2] + matches[:, 2:]) / 2 - priors[:, :2]
    t_cxcy /= priors[:, 2:] 

    # w h target
    t_wh = (matches[:, 2:] - matches[:, :2]) / priors[:, 2:] # HERE. must larger than 0
    t_wh = torch.log(t_wh + 1e-10)

    # cx cy w h target
    locs = torch.cat((t_cxcy, t_wh), dim=1)
    # print(locs.shape, clss.shape)

    return locs, clss


if __name__ == '__main__':

    # box1 = torch.rand(3, 4)
    # box2 = torch.rand(10, 4)
    # inters = bbox_overlap(box1, box2)
    # print(inters.shape)

    priorbox = PriorBox()
    priors = priorbox()

    # tic = time.time()
    # print(priors[8000:8100])
    # print(time.time() - tic)

    gts = priors[:5]
    labels = torch.arange(5)

    locs, clss = match(gts, labels, priors)

    

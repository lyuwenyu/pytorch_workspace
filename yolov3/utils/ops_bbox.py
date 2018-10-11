import numpy as np 
import torch
import math
import time

import torch

class PriorBox(object):
    def __init__(self, ):
        
        self.img_dim = 300
        self.strides = [8, 16, 32]
        self.grids = [int(self.img_dim / s) for s in self.strides]

        # anchor size
        self.base_sizes = [60, 120, 180]
        self.area_rarios = [[2, ], [2, ], [2, ]]
        self.aspect_rarios = [[2, ], [2, 3], [2, 3]]

    def __call__(self, ):
        '''
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
            
            n = xx.shape[0] * xx.shape[1]
            priors = np.zeros((n, len(whs), 4))
            # priors = torch.zeros((n, len(whs), 4))
            priors[:, :, 0] = cx.reshape(-1, 1)
            priors[:, :, 1] = cy.reshape(-1, 1)
            priors[:, :, 2:] = whs

            anchors += [priors.reshape(-1, 4)]
            print(n, len(whs))

        anchors = np.concatenate(anchors, axis=0)
        # anchors = torch.cat(anchors, dim=0)
        print(anchors.shape)
        anchors = torch.from_numpy(anchors)
        # anchors.clamp_(max=1., min=0.)

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

    inter = torch.clamp(max_points-min_points, min=0)

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


def match():
    pass

    

if __name__ == '__main__':

    box1 = torch.rand(3, 4)
    box2 = torch.rand(10, 4)

    inters = bbox_overlap(box1, box2)
    # print(inters.shape)

    priorbox = PriorBox()

    tic = time.time()
    priorbox()
    print(time.time() - tic)
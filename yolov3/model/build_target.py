import torch
import numpy as np


def bbox_iou(box1, box2, x1y1x2y2=True):
    '''
    '''
    if x1y1x2y2:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    
    else:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    x1 = torch.max(b1_x1, b2_x1)
    y1 = torch.max(b1_y1, b2_y1)
    x2 = torch.min(b1_x2, b2_x2)
    y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp((x2 - x1 + 1), min=0)* torch.clamp((y2 - y1 + 1), min=0)

    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area + le-16)

    return iou



def build_target(pred_boxes, pred_conf, pred_cls, target, scaled_anchors, nA, nC, nG, requestPrecision):
    '''
    nGT, nCorrect, tx, ty, tw, th, tconf, tcls
    '''

    nB = len(target)
    nT = [len(bs) for bs in target]

    tx = torch.zeros(nB, nA, nG, nG)
    ty = torch.zeros(nB, nA, nG, nG)
    tw = torch.zeros(nB, nA, nG, nG)
    th = torch.zeros(nB, nA, nG, nG)

    tconf = torch.ByteTensor(nB, nA, nG, nG).fill_(0)
    tcls = torch.ByteTensor(nB, nA, nG, nG, nC).fill_(0)

    TP = torch.ByteTensor(nB, max(nT)).fill_(0)
    FP = torch.ByteTensor(nB, max(nT)).fill_(0)
    FN = torch.ByteTensor(nB, max(nT)).fill_(0)
    TC = torch.ShortTensor(nB, max(nT)).fill_(-1)

    for i in range(nB):

        nTb = nT[i]
        if nTb == 0: continue
        
        t = target[i]  # [[cls, x, y, h, w], [cls, x, y, h, w]]

        tc = t[:, 0].long()
        gx, gy, gw, gh = t[:, 1] * nG, t[:, 2] * nG, t[:, 3] * nG, t[:, 4] * nG

        gi = torch.clamp(gx.long(), min=0, max=nG-1)
        gj = torch.clamp(gy.long(), min=0, max=nG-1)
        
        box_t = t[:, 3: ] * nG # w, h: (n_gt, 2)
        box_a = scaled_anchors.unsqueeze(1).repeat(1, nTb, 1) # w, h: (nA, n_gt, 2)

        inter_area = torch.min(box_t, box_a).prod(2)
        iou_anchor = inter_area / (gw * gh + box_a.prod(2) - inter_area + 1e-15)
        iou_anchor_best, a = iou_anchor.max(0) # best anchor for each target.

        # two targets can not claim the same anchor.
        if nTb > 1:
            _, iou_order = torch.sort(iou_anchor_best, descending=True)
            u = gi.float() * 0.3425405 + gj.float() * 0.2343235 * a.float() * 0.6462432
            uniq = torch.unique(u[iou_order])
            _, uindex = np.unique(u[iou_order], return_index=True) # using numpy function
            # print(type(uindex))
            k = iou_order[uindex]
            k = k[iou_anchor_best[k] > 0.01]
            if len(k) == 0: continue
        
            a = a[k]
            gi = gi[k]
            gj = gj[k]
            gx = gx[k]
            gy = gy[k]
            gw = gw[k]
            gh = gh[k]
            tc = tc[k]

        else:
            if iou_anchor_best < 0.01: continue
            k = 0
        
        tx[i, a, gj, gi] = gx - gi.float()
        ty[i, a, gj, gi] = gy - gj.float()

        tw[i, a, gj, gi] = torch.log(gw / scaled_anchors[a, 0] + 1e-15)
        th[i, a, gj, gi] = torch.log(gh / scaled_anchors[a, 1] + 1e-15)

        tconf[i, a, gj, gi] = 1
        tcls[i, a, gj, gi, tc] = 1

        if requestPrecision:
            pass

    print(tcls.requires_grad)

    return tx, ty, tw, th, tconf, tcls



if __name__ == '__main__':

    target = [torch.rand(4, 5)] * 8
    scaled_anchors = torch.rand(3, 2)

    build_target(0, 0, 0, target, scaled_anchors, 3, 80, 13, False)
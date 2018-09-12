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

    inter_area = torch.clamp((x2 - x1 + 1), min=0) * torch.clamp((y2 - y1 + 1), min=0)

    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
     
    iou = inter_area / (b1_area + b2_area - inter_area + le-16)

    return iou


def _build_target(pred_boxes, pred_conf, pred_cls, target, scaled_anchors, nA, nC, nG, requestPrecision):
    '''
    nGT, nCorrect, tx, ty, tw, th, tconf, tcls
    '''
    ignore_threshold = 0.5

    nB = len(target)
    nT = [len(bs) for bs in target]

    tx = torch.zeros(nB, nA, nG, nG)
    ty = torch.zeros(nB, nA, nG, nG)
    tw = torch.zeros(nB, nA, nG, nG)
    th = torch.zeros(nB, nA, nG, nG)

    tconf = torch.ByteTensor(nB, nA, nG, nG).fill_(0)
    tcls = torch.ByteTensor(nB, nA, nG, nG, nC).fill_(0)

    conf_mask = torch.ByteTensor(nB, nA, nG, nG).fill_(1)

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
        
        box_t = t[:, 3: ] * nG  # w, h: (n_gt, 2)
        box_a = scaled_anchors.unsqueeze(1).repeat(1, nTb, 1)  # w, h: (nA, n_gt, 2)

        inter_area = torch.min(box_t, box_a).prod(2)
        iou_anchor = inter_area / (gw * gh + box_a.prod(2) - inter_area + 1e-15)

        # ignore overlap > threshhold
        _gi = gi.repeat(nA, 1)[iou_anchor > ignore_threshold]
        _gj = gj.repeat(nA, 1)[iou_anchor > ignore_threshold]
        _a = torch.arange(nA).view(nA, 1).repeat(1, nTb).long()[iou_anchor > ignore_threshold]
        conf_mask[i, _a, _gj, _gi] = 0

        iou_anchor_best, a = iou_anchor.max(0) # best anchor for each target.

        # two targets can not claim the same anchor.
        if nTb > 1:
            _, iou_order = torch.sort(iou_anchor_best, descending=True)
            u = gi.float() * 0.3425405 + gj.float() * 0.2343235 * a.float() * 0.6462432
            # uniq = torch.unique(u[iou_order])
            # _, uindex = np.unique(u[iou_order], return_index=True) # using numpy function

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

        ### one box prior for each gt.

        tx[i, a, gj, gi] = gx - gi.float()
        ty[i, a, gj, gi] = gy - gj.float()

        tw[i, a, gj, gi] = torch.log(gw / scaled_anchors[a, 0] + 1e-15)
        th[i, a, gj, gi] = torch.log(gh / scaled_anchors[a, 1] + 1e-15)

        tconf[i, a, gj, gi] = 1
        tcls[i, a, gj, gi, tc] = 1
        conf_mask[i, a, gj, gi] = 1

        if requestPrecision:
            pass

    return tx, ty, tw, th, tconf, tcls, conf_mask


def build_target(pred_boxes, pred_conf, pred_cls, target, scaled_anchors, nA, nC, nG, requestPrecision):
    '''
    target: n, t, [c, x, y, w, h]

    nGT, nCorrect, tx, ty, tw, th, tconf, tcls
    '''
    ignore_threshold = 0.5

    nB = len(target)
    nT = [len(bs) for bs in target]
    maxnT = max(nT)

    tx = torch.zeros(nB, nA, nG, nG).to(device=pred_boxes.device)
    ty = torch.zeros(nB, nA, nG, nG).to(device=pred_boxes.device)
    tw = torch.zeros(nB, nA, nG, nG).to(device=pred_boxes.device)
    th = torch.zeros(nB, nA, nG, nG).to(device=pred_boxes.device)

    tcls = torch.zeros(nB, nA, nG, nG, nC).to(device=pred_boxes.device) #.fill_(0)
    tconf = torch.ByteTensor(nB, nA, nG, nG).fill_(0).to(device=pred_boxes.device)

    conf_mask = torch.ByteTensor(nB, nA, nG, nG).fill_(1).to(device=pred_boxes.device)

    TP = torch.ByteTensor(nB, maxnT).fill_(0)
    FP = torch.ByteTensor(nB, maxnT).fill_(0)
    FN = torch.ByteTensor(nB, maxnT).fill_(0)
    TC = torch.ShortTensor(nB, maxnT).fill_(-1)

    nGt = 0

    for b in range(nB):
        for t in range(nT[b]):
            # if target[b, t].sum() == 0:
            #     continue
            # print(b, t)

            nGt += 1
            
            tc = target[b][t, 0].long()
            gx = target[b][t, 1] * nG
            gy = target[b][t, 2] * nG
            gw = target[b][t, 3] * nG
            gh = target[b][t, 4] * nG

            gi = torch.clamp(gx.long(), min=0, max=nG-1)
            gj = torch.clamp(gy.long(), min=0, max=nG-1)

            inter_area = torch.min(target[b][t, 3: ].view(1, 2), scaled_anchors).prod(1)
            iou_anchor = inter_area / (gw * gh + target[b][t, 3: ].prod() - inter_area + 1e-15)

            conf_mask[b, iou_anchor > ignore_threshold, gj, gi] = 0
            _, a = iou_anchor.max(0) # best anchor for target.

            tx[b, a, gj, gi] = gx - gi.float()
            ty[b, a, gj, gi] = gy - gj.float()

            tw[b, a, gj, gi] = torch.log(gw / scaled_anchors[a, 0] + 1e-15)
            th[b, a, gj, gi] = torch.log(gh / scaled_anchors[a, 1] + 1e-15)

            tconf[b, a, gj, gi] = 1
            tcls[b, a, gj, gi, tc] = 1
            conf_mask[b, a, gj, gi] = 1
            
    return tx, ty, tw, th, tconf, tcls, conf_mask



if __name__ == '__main__':

    target = torch.rand(2, 2, 5)
    scaled_anchors = torch.rand(3, 2)

    build_target(0, 0, 0, target, scaled_anchors, 3, 80, 13, False)
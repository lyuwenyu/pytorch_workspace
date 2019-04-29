import torch
import numpy as np
import random


def _build_target(pred_boxes, pred_conf, pred_cls, target, scaled_anchors, nA, nC, nG, requestPrecision):
    '''
    nGT, nCorrect, tx, ty, tw, th, tconf, tcls
    '''
    ignore_threshold = 0.3

    nB = len(target)
    nT = [len(bs) for bs in target]

    tx = torch.zeros(nB, nA, nG, nG).to(device=pred_boxes.device)
    ty = torch.zeros(nB, nA, nG, nG).to(device=pred_boxes.device)
    tw = torch.zeros(nB, nA, nG, nG).to(device=pred_boxes.device)
    th = torch.zeros(nB, nA, nG, nG).to(device=pred_boxes.device)

    tconf = torch.ByteTensor(nB, nA, nG, nG).fill_(0).to(device=pred_boxes.device)
    tcls = torch.zeros(nB, nA, nG, nG, nC).to(device=pred_boxes.device)

    conf_mask = torch.ByteTensor(nB, nA, nG, nG).fill_(-1).to(device=pred_boxes.device)

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
        
        box_t = t[:, 1: ] * nG  # w, h: (n_gt, 2)
        box_a = scaled_anchors.unsqueeze(1).repeat(1, nTb, 1)  # w, h: (nA, n_gt, 2)
        overlap = bbox_overlap(box_t, box_a, x1y1x2y2=False)

        inter_area = torch.min(box_t, box_a).prod(2)
        iou_anchor = inter_area / (gw * gh + box_a.prod(2) - inter_area + 1e-15)

        # ignore 
        _ignore = iou_anchor > ignore_threshold
        # _a = torch.arange(nA).view(3, 1).repeat(1, nTb).to(dtype=torch.long, device=pred_boxes.device)[_ignore]
        # _gi = gi.repeat(nA, 1)[_ignore]
        # _gj = gj.repeat(nA, 1)[_ignore]
        # conf_mask[i, _a, _gj, _gi] = 0
        # print(iou_anchor.shape)
        # print(_ignore.max(0), _ignore.max(0).shape)
        # print(_ignore.max(1), _ignore.max(1).shape)
        # c += 1

        conf_mask[i, _ignore.max(0), gj[_ignore.max(1)], gi[_ignore.max(1)]]

        _, a = iou_anchor.max(0) # best anchor for each target.
        
        # two targets can not claim the same anchor.

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
    # maxnT = max(nT)

    tx = torch.zeros(nB, nA, nG[0], nG[1]).to(device=pred_boxes.device)
    ty = torch.zeros(nB, nA, nG[0], nG[1]).to(device=pred_boxes.device)
    tw = torch.zeros(nB, nA, nG[0], nG[1]).to(device=pred_boxes.device)
    th = torch.zeros(nB, nA, nG[0], nG[1]).to(device=pred_boxes.device)

    tcls = torch.zeros(nB, nA, nG[0], nG[1], nC).to(device=pred_boxes.device) #.fill_(0)
    tconf = torch.ByteTensor(nB, nA, nG[0], nG[1]).fill_(0).to(device=pred_boxes.device) # mask

    # conf_mask = torch.zeros(nB, nA, nG, nG).fill_(0).to(device=pred_boxes.device)
    conf_mask = torch.zeros(nB, nA, nG[0], nG[1]).fill_(-1).to(device=pred_boxes.device)

    # TP = torch.ByteTensor(nB, maxnT).fill_(0)
    # FP = torch.ByteTensor(nB, maxnT).fill_(0)
    # FN = torch.ByteTensor(nB, maxnT).fill_(0)
    # TC = torch.ShortTensor(nB, maxnT).fill_(-1)

    nGt = 0

    for b in range(nB):

        # used = torch.zeros(nT[b]).fill_(-1).to(device=pred_boxes.device)
        # bbx_gt_index = range(nT[b])
        # random.shuffle(bbx_gt_index)
        # for t in range(bbx_gt_index):

        for t in range(nT[b]):

            tc = target[b][t, 0].long()
            gx = target[b][t, 1] * nG[1]
            gy = target[b][t, 2] * nG[0]
            gw = target[b][t, 3] * nG[1]
            gh = target[b][t, 4] * nG[0]
            # gbox = target[b][t, 3:] * nG
            gbox = target[b][t, 3:]
            gbox[0] *= nG[1]
            gbox[1] *= nG[0]

            gi = torch.clamp(gx.long(), min=0, max=nG[1] - 1)
            gj = torch.clamp(gy.long(), min=0, max=nG[0] - 1)

            inter_area = torch.min(gbox, scaled_anchors).prod(1)
            iou_anchor = inter_area / (gbox.prod(0) + scaled_anchors.prod(1) - inter_area + 1e-15)

            # print(iou_anchor)
            # c+=1

            # conf_mask[b, iou_anchor < ignore_threshold] = -1
            conf_mask[b, iou_anchor > ignore_threshold, gj, gi] = 0

            # one target to one anchor exclusively.
            _, a = iou_anchor.max(0) # best anchor for target.

            # ious, aindex = torch.sort(iou_anchor, descending=True)
            # a = -1
            # for ai, iou in zip(aindex, ious):
            #     uid = gj.float() * 0.32432533 + gi.float() * 0.53243245 + ai.float() * 0.63321341
            #     if (uid == used).sum() == 0 and iou > 0.5:
            #         used[t] = uid
            #         a = ai
            #         break
            #     else:
            #         # print((uid == used).sum())
            #         # print('---using same anchor for one target---')
            #         pass  
            # if a == -1: # or used[t] == -1
            #     # print('-----no anchor for this target------')
            #     continue

            tx[b, a, gj, gi] = gx - gi.float()
            ty[b, a, gj, gi] = gy - gj.float()

            tw[b, a, gj, gi] = torch.log(gw / scaled_anchors[a, 0] + 1e-15)
            th[b, a, gj, gi] = torch.log(gh / scaled_anchors[a, 1] + 1e-15)

            tcls[b, a, gj, gi, tc] = 1
            tconf[b, a, gj, gi] = 1
            conf_mask[b, a, gj, gi] = 1

    return tx, ty, tw, th, tconf, tcls, conf_mask


def build_quad_target(pred_boxes, target, scaled_anchors, nA, nC, nG, requestPrecision):
    '''
    target: n, t, [c, x, y, w, h]

    nGT, nCorrect, tx, ty, tw, th, tconf, tcls
    '''
    ignore_threshold = 0.5

    nB = len(target)
    nT = [len(bs) for bs in target]
    # maxnT = max(nT)

    tx = torch.zeros(nB, nA, nG, nG).to(device=pred_boxes.device)
    ty = torch.zeros(nB, nA, nG, nG).to(device=pred_boxes.device)
    tw = torch.zeros(nB, nA, nG, nG).to(device=pred_boxes.device)
    th = torch.zeros(nB, nA, nG, nG).to(device=pred_boxes.device)

    txp1 = torch.zeros(nB, nA, nG, nG).to(dtype=pred_boxes.dtype, device=pred_boxes.device)
    typ1 = torch.zeros(nB, nA, nG, nG).to(dtype=pred_boxes.dtype, device=pred_boxes.device)
    txp2 = torch.zeros(nB, nA, nG, nG).to(dtype=pred_boxes.dtype, device=pred_boxes.device)
    typ2 = torch.zeros(nB, nA, nG, nG).to(dtype=pred_boxes.dtype, device=pred_boxes.device)
    txp3 = torch.zeros(nB, nA, nG, nG).to(dtype=pred_boxes.dtype, device=pred_boxes.device)
    typ3 = torch.zeros(nB, nA, nG, nG).to(dtype=pred_boxes.dtype, device=pred_boxes.device)
    txp4 = torch.zeros(nB, nA, nG, nG).to(dtype=pred_boxes.dtype, device=pred_boxes.device)
    typ4 = torch.zeros(nB, nA, nG, nG).to(dtype=pred_boxes.dtype, device=pred_boxes.device)

    tcls = torch.zeros(nB, nA, nG, nG, nC).to(device=pred_boxes.device) #.fill_(0)
    # print(tcls.shape)
    tconf = torch.ByteTensor(nB, nA, nG, nG).fill_(0).to(device=pred_boxes.device) # mask

    # conf_mask = torch.zeros(nB, nA, nG, nG).fill_(0).to(device=pred_boxes.device)
    conf_mask = torch.zeros(nB, nA, nG, nG).fill_(-1).to(device=pred_boxes.device)

    nGt = 0

    for b in range(nB):

        for t in range(nT[b]):

            tc = target[b][t, 0].long()
            p1 = target[b][t, [1, 2]] * nG
            p2 = target[b][t, [3, 4]] * nG
            p3 = target[b][t, [5, 6]] * nG
            p4 = target[b][t, [7, 8]] * nG

            xmin = torch.min(target[b][t, [1, 3, 5, 7]]) * nG
            xmax = torch.max(target[b][t, [1, 3, 5, 7]]) * nG
            ymin = torch.min(target[b][t, [2, 4, 6, 8]]) * nG
            ymax = torch.max(target[b][t, [2, 4, 6, 8]]) * nG

            # assert xmin == p1[0], ''
            # assert xmax == p3[0], ''
            # assert ymin == p1[1], ''
            # assert ymax == p3[1], ''

            gx = (xmin + xmax) / 2.
            gy = (ymin + ymax) / 2.
            gw = xmax - xmin
            gh = ymax - ymin

            # gbox = torch.cat((gw.view(1, -1), gh.view(1, -1)), dim=-1).view(2) # target[b][t, 3:] * nG
            # print(gbox.shape)
            gi = torch.clamp(gx.long(), min=0, max=nG - 1)
            gj = torch.clamp(gy.long(), min=0, max=nG - 1)

            inter_area = torch.min(gw, scaled_anchors[:, 0]) * torch.min(gh, scaled_anchors[:, 1])
            iou_anchor = inter_area / (gw * gh + scaled_anchors.prod(1) - inter_area + 1e-15)

            # conf_mask[b, iou_anchor < ignore_threshold] = -1
            conf_mask[b, iou_anchor > ignore_threshold, gj, gi] = 0

            # one target to one anchor exclusively.
            _, a = iou_anchor.max(0) # best anchor for target.

            tx[b, a, gj, gi] = gx - gi.float()
            ty[b, a, gj, gi] = gy - gj.float()

            tw[b, a, gj, gi] = torch.log(gw / scaled_anchors[a, 0] + 1e-15)
            th[b, a, gj, gi] = torch.log(gh / scaled_anchors[a, 1] + 1e-15)

            # anchor clockwise points, bingxing TODO
            # aw = scaled_anchors[a, 0]
            # ah = scaled_anchors[a, 1]
            # ap1x = gi.float() - aw / 2
            # ap1y = gj.float() - ah / 2
            # ap2x = gi.float() + aw / 2
            # ap2y = gj.float() - ah / 2
            # ap3x = gi.float() + aw / 2
            # ap3y = gj.float() + ah / 2            
            # ap4x = gi.float() - aw / 2
            # ap4y = gj.float() + ah / 2 
            # txp1[b, a, gj, gi] = (p1[0] - ap1x) / aw
            # typ1[b, a, gj, gi] = (p1[1] - ap1y) / ah
            # txp2[b, a, gj, gi] = (p2[0] - ap2x) / aw
            # typ2[b, a, gj, gi] = (p2[1] - ap2y) / ah
            # txp3[b, a, gj, gi] = (p3[0] - ap3x) / aw
            # typ3[b, a, gj, gi] = (p3[1] - ap3y) / ah
            # txp4[b, a, gj, gi] = (p4[0] - ap4x) / aw
            # typ4[b, a, gj, gi] = (p4[1] - ap4y) / ah

            # # pred bbox points, chuanxing TODO
            cx = pred_boxes[b, a, gj, gi, 0]
            cy = pred_boxes[b, a, gj, gi, 1]
            aw = pred_boxes[b, a, gj, gi, 2]
            ah = pred_boxes[b, a, gj, gi, 3]
            # ap1x = cx - aw / 2
            # ap1y = cy - ah / 2
            # ap2x = cx + aw / 2
            # ap2y = cy - ah / 2
            # ap3x = cx + aw / 2
            # ap3y = cy + ah / 2
            # ap4x = cx - aw / 2
            # ap4y = cy + ah / 2 
            # # txp1[b, a, gj, gi] = (p1[0] - ap1x) / aw
            # # typ1[b, a, gj, gi] = (p1[1] - ap1y) / ah
            # # txp2[b, a, gj, gi] = (p2[0] - ap2x) / aw
            # # typ2[b, a, gj, gi] = (p2[1] - ap2y) / ah
            # # txp3[b, a, gj, gi] = (p3[0] - ap3x) / aw
            # # typ3[b, a, gj, gi] = (p3[1] - ap3y) / ah
            # # txp4[b, a, gj, gi] = (p4[0] - ap4x) / aw
            # # typ4[b, a, gj, gi] = (p4[1] - ap4y) / ah
            # txp1[b, a, gj, gi] = (p1[0] - ap1x) / scaled_anchors[a, 0]
            # typ1[b, a, gj, gi] = (p1[1] - ap1y) / scaled_anchors[a, 1]
            # txp2[b, a, gj, gi] = (p2[0] - ap2x) / scaled_anchors[a, 0]
            # typ2[b, a, gj, gi] = (p2[1] - ap2y) / scaled_anchors[a, 1]
            # txp3[b, a, gj, gi] = (p3[0] - ap3x) / scaled_anchors[a, 0]
            # typ3[b, a, gj, gi] = (p3[1] - ap3y) / scaled_anchors[a, 1]
            # txp4[b, a, gj, gi] = (p4[0] - ap4x) / scaled_anchors[a, 0]
            # typ4[b, a, gj, gi] = (p4[1] - ap4y) / scaled_anchors[a, 1]

            # pred bbox points, chuanxing TODO
            aw = scaled_anchors[a, 0]
            ah = scaled_anchors[a, 1]
            txp1[b, a, gj, gi] = (gx - p1[0])  
            typ1[b, a, gj, gi] = (gy - p1[1]) 
            txp2[b, a, gj, gi] = (gx - p2[0]) 
            typ2[b, a, gj, gi] = (gy - p2[1]) 
            txp3[b, a, gj, gi] = (gx - p3[0]) 
            typ3[b, a, gj, gi] = (gy - p3[1]) 
            txp4[b, a, gj, gi] = (gx - p4[0]) 
            typ4[b, a, gj, gi] = (gy - p4[1]) 

            # target 
            tcls[b, a, gj, gi, tc] = 1
            tconf[b, a, gj, gi] = 1
            conf_mask[b, a, gj, gi] = 1

    return tx, ty, tw, th, txp1, typ1, txp2, typ2, txp3, typ3, txp4, typ4, tconf, tcls, conf_mask




if __name__ == '__main__':

    target = torch.rand(2, 2, 5)
    scaled_anchors = torch.rand(3, 2)



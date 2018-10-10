import torch
import torch.nn as nn
import numpy as np

from build_target import build_target

VERSION = torch.__version__

#---

class YOLOLayer(nn.Module):
    def __init__(self, anchors, num_classes, anchor_index, img_dim=None):
        super(YOLOLayer, self).__init__()
        
        self.anchors = anchors
        self.nA = len(anchors)
        self.nC = num_classes

        self.bbox_attrs = num_classes + 5

        self.max_grid = 100

        if anchor_index[0] == 6:
            stride = 32.
        elif anchor_index[0] == 3:
            stride = 16.
        else:
            stride = 8.

        # self.max_grid = 100 # int(img_dim / stride)
        self.stride = stride

        self.scaled_anchors = torch.tensor(anchors) / stride   
        self.anchor_w = self.scaled_anchors[:, 0]
        self.anchor_h = self.scaled_anchors[:, 1]

        if VERSION >= '0.4.1':
            self.grid_h, self.grid_w = torch.meshgrid((torch.arange(self.max_grid), torch.arange(self.max_grid)))
        else:
            self.grid_w = torch.arange(self.max_grid).repeat(self.max_grid, 1)
            self.grid_h = torch.arange(self.max_grid).repeat(self.max_grid, 1).t()

        self.mseLoss = nn.MSELoss() # size_average=True
        self.bceLoss = nn.BCEWithLogitsLoss()
        self.crossentropy = nn.CrossEntropyLoss()

    def forward(self, p, target=None, requestPrecision=False, epoch=None):
        '''forward '''
        bs = p.shape[0]
        nG = p.shape[2]
        grid_x = self.grid_w[:nG, :nG].to(dtype=p.dtype, device=p.device)
        grid_y = self.grid_h[:nG, :nG].to(dtype=p.dtype, device=p.device)

        p = p.view(bs, self.nA, self.bbox_attrs, nG, nG).permute(0, 1, 3, 4, 2).contiguous()

        x = torch.sigmoid(p[..., 0])
        y = torch.sigmoid(p[..., 1])
        w = p[..., 2]
        h = p[..., 3]

        width = torch.exp(w) * self.anchor_w.view(1, -1, 1, 1).to(dtype=w.dtype, device=w.device) #.data
        height = torch.exp(h) * self.anchor_h.view(1, -1, 1, 1).to(dtype=h.dtype, device=h.device)

        pred_boxes = torch.zeros((bs, self.nA, nG, nG, 4)).to(dtype=p.dtype, device=p.device)
        
        pred_conf = p[..., 4] 
        pred_cls =  p[..., 5:] 

        # cross entropy
        # pred_cls = p[..., 5:]

        if target is None: # inference phase
            pred_boxes[..., 0] = x + grid_x.float()
            pred_boxes[..., 1] = y + grid_y.float()
            pred_boxes[..., 2] = width
            pred_boxes[..., 3] = height

            out = torch.cat((
                pred_boxes.view(bs, -1, 4) * self.stride,
                torch.sigmoid(pred_conf).view(bs, -1, 1),
                torch.sigmoid(pred_cls).view(bs, -1, self.nC)),
                # nn.functional.softmax(pred_cls, dim=-1).view(bs, -1, self.nC)), 
                dim=-1)

            return out
        
        else: # train phase TODO 

            tx, ty, tw, th, tconf, tcls, conf_mask = build_target(pred_boxes.data, # here 
                                                            pred_conf.data, 
                                                            pred_cls.data, 
                                                            target, 
                                                            self.scaled_anchors.to(device=p.device), 
                                                            self.nA, self.nC, nG,
                                                            requestPrecision)
            numobj = tconf.sum().float()
            mask = tconf

            if numobj > 0:

                # pred_conf = torch.sigmoid(pred_conf)
                # pred_cls =  torch.sigmoid(pred_cls)

                lx = self.mseLoss(x[mask], tx[mask])
                ly = self.mseLoss(y[mask], ty[mask])
                lw = self.mseLoss(w[mask], tw[mask])
                lh = self.mseLoss(h[mask], th[mask])

                # lconf = self.bceLoss(pred_conf[conf_mask != 0], tconf[conf_mask != 0].to(dtype=pred_conf.dtype))
                lconf_bg = self.bceLoss(pred_conf[conf_mask == -1], tconf[conf_mask == -1].to(dtype=pred_conf.dtype))
                lconf_ob = self.bceLoss(pred_conf[conf_mask == 1], tconf[conf_mask == 1].to(dtype=pred_conf.dtype))
                lconf = 2 * lconf_bg + lconf_ob # 2. * lconf_bg + lconf_ob

                lcls = self.bceLoss(pred_cls[mask], tcls[mask])
                # lcls = self.crossentropy(pred_cls[mask], tcls[mask].argmax(1))

            else:
                lx, ly, lw, lh, lcls, lconf = [torch.tensor(0.).to(dtype=torch.float32, device=p.device)] * 6

            loss = (lx + ly + lw + lh + lconf + lcls) * numobj / (bs + 1e-8)
            
            # writer.add_scalars('losses', {
            #     'loss': loss.item(),
            #     'lx': lx.item(),
            #     'ly': ly.item(),
            #     'lw': lw.item(),
            #     'lh': lh.item(),
            #     'lcong': lconf.item(),
            #     'lcls': lcls.item()
            # })

            return loss, lx, ly, lw, lh, lconf, lcls


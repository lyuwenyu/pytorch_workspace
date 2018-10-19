import torch
import torch.nn as nn

from build_target import build_quad_target

VERSION = torch.__version__

class textbox_layer(nn.Module):
    def __init__(self, anchors, num_classes, anchor_index, img_dim=None):
        super(textbox_layer, self).__init__()
        
        self.anchors = anchors
        self.nA = len(anchors)
        self.nC = num_classes
 
        self.bbox_attrs = num_classes + 4 + 8 + 1

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
        self.smoothl1loss = nn.SmoothL1Loss()

    def forward(self, p, target=None, requestPrecision=False, epoch=None):
        '''forward '''
        bs = p.shape[0]
        nG = p.shape[2]
        grid_x = self.grid_w[:nG, :nG].to(dtype=p.dtype, device=p.device)
        grid_y = self.grid_h[:nG, :nG].to(dtype=p.dtype, device=p.device)
        anchor_w = self.anchor_w.view(1, -1, 1, 1).to(dtype=p.dtype, device=p.device)
        anchor_h = self.anchor_h.view(1, -1, 1, 1).to(dtype=p.dtype, device=p.device)

        p = p.view(bs, self.nA, self.bbox_attrs, nG, nG).permute(0, 1, 3, 4, 2).contiguous()

        x = torch.sigmoid(p[..., 0])
        y = torch.sigmoid(p[..., 1])
        w = p[..., 2]
        h = p[..., 3]

        # TODO for qaud points offset

        p1x = p[..., 4]
        p1y = p[..., 5]
        p2x = p[..., 6]
        p2y = p[..., 7]
        p3x = p[..., 8]
        p3y = p[..., 9]
        p4x = p[..., 10]
        p4y = p[..., 11]
        # p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y = p[..., [4, 5, 6, 7, 8, 9, 10, 11]]

        width = torch.exp(w) * anchor_w #.data
        height = torch.exp(h) * anchor_h

        pred_boxes = torch.zeros((bs, self.nA, nG, nG, 4)).to(dtype=p.dtype, device=p.device)
        pred_vertex = torch.zeros((bs, self.nA, nG, nG, 8)).to(dtype=p.dtype, device=p.device)

        pred_conf = p[..., 12] 
        pred_cls =  p[..., 13:]

        if target is None: # inference phase
            pred_boxes[..., 0] = x + grid_x.float()
            pred_boxes[..., 1] = y + grid_y.float()
            pred_boxes[..., 2] = width
            pred_boxes[..., 3] = height

            # TODO using which vertex is a problem here.
            pred_vertex[..., 0] = p1x * anchor_w + (grid_x.float() - anchor_w / 2) 
            pred_vertex[..., 1] = p1y * anchor_h + (grid_y.float() - anchor_h / 2 )
            pred_vertex[..., 2] = p2x * anchor_w + (grid_x.float() + anchor_w / 2) 
            pred_vertex[..., 3] = p2y * anchor_h + (grid_y.float() - anchor_h / 2 )
            pred_vertex[..., 4] = p3x * anchor_w + (grid_x.float() + anchor_w / 2) 
            pred_vertex[..., 5] = p3y * anchor_h + (grid_y.float() + anchor_h / 2 )
            pred_vertex[..., 6] = p4x * anchor_w + (grid_x.float() - anchor_w / 2) 
            pred_vertex[..., 7] = p4y * anchor_h + (grid_y.float() + anchor_h / 2 )

            out = torch.cat((
                pred_boxes.view(bs, -1, 4) * self.stride,
                pred_vertex.view(bs, -1, 8) * self.stride,
                torch.sigmoid(pred_conf).view(bs, -1, 1),
                torch.sigmoid(pred_cls).view(bs, -1, self.nC)),
                dim=-1)

            return out
        
        else: # train phase TODO 

            tx, ty, tw, th, txp1, typ1, txp2, typ2, txp3, typ3, txp4, typ4, tconf, tcls, conf_mask = build_quad_target(
                                                            pred_boxes.data, # here 
                                                            target, 
                                                            self.scaled_anchors.to(device=p.device), 
                                                            self.nA, self.nC, nG,
                                                            requestPrecision)
            
            numobj = tconf.sum().float()
            mask = tconf

            if numobj > 0:

                lx = self.mseLoss(x[mask], tx[mask])
                ly = self.mseLoss(y[mask], ty[mask])
                lw = self.mseLoss(w[mask], tw[mask])
                lh = self.mseLoss(h[mask], th[mask])

                # lconf = self.bceLoss(pred_conf[conf_mask != 0], tconf[conf_mask != 0].to(dtype=pred_conf.dtype))
                lconf_bg = self.bceLoss(pred_conf[conf_mask == -1], tconf[conf_mask == -1].to(dtype=pred_conf.dtype))
                lconf_ob = self.bceLoss(pred_conf[conf_mask == 1], tconf[conf_mask == 1].to(dtype=pred_conf.dtype))
                lconf = lconf_bg + lconf_ob # 2. * lconf_bg + lconf_ob

                lcls = self.bceLoss(pred_cls[mask], tcls[mask])
                # lcls = self.crossentropy(pred_cls[mask], tcls[mask].argmax(1))

                lp1x = self.mseLoss(p1x[mask], txp1[mask]) # self.smoothl1loss
                lp1x = self.mseLoss(p1y[mask], typ1[mask])
                lp2x = self.mseLoss(p2x[mask], txp2[mask])
                lp2y = self.mseLoss(p2y[mask], typ2[mask])
                lp3x = self.mseLoss(p3x[mask], txp3[mask])
                lp3y = self.mseLoss(p3y[mask], typ3[mask])
                lp4x = self.mseLoss(p4x[mask], txp4[mask])
                lp4y = self.mseLoss(p4y[mask], typ4[mask])

            else:
                lx, ly, lw, lh, lcls, lconf = [torch.tensor(0.).to(dtype=torch.float32, device=p.device)] * 6
                lp1x, lp1x, lp2x, lp2y, lp3x, lp3y, lp4x, lp4y = [torch.tensor(0.).to(dtype=torch.float32, device=p.device)] * 8

            # loss1 = (lx + ly + lw + lh + lconf + lcls) * numobj / (bs + 1e-8)
            loss = (lx + ly + lw + lh + lconf + lcls + lp1x + lp1x + lp2x + lp2y + lp3x + lp3y + lp4x + lp4y) * numobj / (bs + 1e-8)


            return loss, lx, ly, lw, lh, lp1x, lp1x, lp2x, lp2y, lp3x, lp3y, lp4x, lp4y, lconf, lcls


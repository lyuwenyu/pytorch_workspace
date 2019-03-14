import torch

def compute_giou(bboxa, bboxb):
    '''
    bboxa: n x 4
    bbox: [x1 y1 x2 y2]
    '''
    ix1 = torch.max(bboxa[:, 0], bboxb[:, 0])
    iy1 = torch.max(bboxa[:, 1], bboxb[:, 1])
    ix2 = torch.min(bboxa[:, 2], bboxb[:, 2])
    iy2 = torch.min(bboxa[:, 3], bboxb[:, 3])
    inter = torch.clamp((ix2 - ix1), min=0) * torch.clamp((iy2 - iy1), min=0)
    
    areaa = (bboxa[:, 2] - bboxa[:, 0]) * (bboxa[:, 3] - bboxa[:, 1])
    areab = (bboxb[:, 2] - bboxb[:, 0]) * (bboxb[:, 3] - bboxb[:, 1])
    union = areaa + areab - inter
    assert (union.data < 0).sum() == 0, ''
    
    cx1 = torch.min(bboxa[:, 0], bboxb[:, 0])
    cy1 = torch.min(bboxa[:, 1], bboxb[:, 1])
    cx2 = torch.max(bboxa[:, 2], bboxb[:, 2])
    cy2 = torch.max(bboxa[:, 3], bboxb[:, 3])
    areac = (cx2 - cx1) * (cy2 - cy1)
    assert (areac.data < 0).sum() == 0, ''
    
    giou = inter / union - (1 - union / areac)
    
    return giou




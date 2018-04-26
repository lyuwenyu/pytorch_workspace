import numpy as np 
import torch
import torch.nn as nn
from torch.autograd import Variable
# from skimage import io, transform
from PIL import Image
from torchvision import transforms


def predict_transform(prediction, inp_dim, anchors, num_classes):

    batch_size = prediction.size()[0]
    stride = inp_dim // prediction.size()[-1]
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    
    # print(batch_size, grid_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    _prediction = Variable(torch.zeros_like(prediction.data))  ## 0.4
    _prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    _prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    _prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
    _prediction[:,:,5:] = torch.sigmoid(prediction[:,:,5:])

    # center_x, center_y
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)
    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)
    x_y_offset = torch.cat([x_offset, y_offset], dim=1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)
    _prediction[:,:,:2] = _prediction[:,:,:2] + Variable(x_y_offset)  ## 0.4

    ## h w 
    anchors = torch.FloatTensor(anchors)
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    _prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4]) * Variable(anchors)  ## 0.4

    _prediction[:,:,:4] = _prediction[:,:,:4] * stride

    return _prediction
    


def get_results(prediction, confidence=0.8, num_classes=80, nms_conf=0.6):
    
    mask = (prediction[:,:,4]>confidence).float().unsqueeze(2)
    prediction = prediction * mask

    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = prediction[:,:,0] - prediction[:,:,2]/2
    box_corner[:,:,1] = prediction[:,:,1] - prediction[:,:,3]/2
    box_corner[:,:,2] = prediction[:,:,0] + prediction[:,:,2]/2
    box_corner[:,:,3] = prediction[:,:,0] + prediction[:,:,3]/2
    prediction[:,:,:4] = box_corner[:,:,:4]


    batch_size = prediction.size()[0]
    
    out = []

    for ind in range(batch_size):

        image_pred = prediction[ind]

        max_conf, max_conf_score = torch.max(image_pred[:,5:], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)        

        seq = (image_pred[:, :5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)

        non_zero_ind = torch.nonzero(image_pred[:,4])
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)
        except:
            continue 
        
        img_classes = unique(image_pred_[:, -1])

        for clss in img_classes:

            cls_mask = image_pred_ * (image_pred_[:, -1] == clss).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)

            conf_sort_index = torch.sort(image_pred_class[:,4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]    
            idx = image_pred_class.size()[0]

            for i in range(idx):
                
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                except:
                    print('\n---ious---\n')
                    break

                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1: ] = image_pred_class[i+1: ] * iou_mask

                non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)


            batch_ind = image_pred_class.new(image_pred_class.size()[0], 1).fill_(ind)
            seq = torch.cat([batch_ind, image_pred_class], 1) # batch_ind, image_pred_class

            out += [ seq ]

    res = torch.cat(out, dim=0)

    return res




def unique(tensor): # 0.4
    
    tnp = tensor.cpu().numpy()
    tnp = np.unique(tnp)
    tnp = torch.from_numpy(tnp)

    tensor_ = tensor.new(tnp.shape)
    tensor_.copy_(tnp)

    return tensor_


def bbox_iou(box1, box2):

    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    inter_area = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)

    iou = inter_area / ( b1_area + b2_area - inter_area)

    return iou




def pre_image(img, inp_dim):
    totensor = transforms.ToTensor()
    img = Image.open(img).resize([inp_dim, inp_dim])
    return totensor(img).unsqueeze(0)


def load_classes(namesfile):
    with open(namesfile, 'r') as f:
        names = f.read().split('\n')[:-1]
    return names
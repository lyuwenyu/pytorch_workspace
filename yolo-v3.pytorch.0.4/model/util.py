import numpy as np 
import torch
import torch.nn as nn


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

    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

    # center_x, center_y
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)
    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)
    x_y_offset = torch.cat([x_offset, y_offset], dim=1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)
    prediction[:,:,:2] += x_y_offset

    ## h w 
    anchors = torch.FloatTensor(anchors)
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4]) * anchors

    prediction[:,:,5:] = torch.sigmoid(prediction[:,:,5:])
    prediction[:,:,:4] *= stride

    return prediction
    
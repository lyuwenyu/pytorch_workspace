import torch
import numpy as np 

from utils import ops_transform

def nms_per_class(bboxes, scores, iou_threshold):
    ''' 
    bbox: dets, 'xyxy'
    scores: conf
    '''
    assert len(bboxes) > 0 and len(bboxes) == len(scores), ''

    index = np.argsort(scores)[::-1]
    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])

    keep = [] 
    while len(index) > 0:
        
        keep += [index[0]]

        _bb = bboxes[index[0]]
        _bbxes = bboxes[index[1: ]]
        x1 = np.maximum(_bb[0], _bbxes[:, 0])
        y1 = np.maximum(_bb[1], _bbxes[:, 1])
        x2 = np.minimum(_bb[2], _bbxes[:, 2])
        y2 = np.minimum(_bb[3], _bbxes[:, 3])
        over = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        ious = over / (areas[index[0]] + areas[index[1: ]] - over)

        _index = np.where(ious < iou_threshold)[0] + 1 # because 0/max_conf
        index = index[_index]

    return np.array(keep)

        
def NMS(dets, objectness_threshold=0., class_threshold=0.3, iou_threshold=0.,):
    '''
    dets: [indictor objectnest, bbox, class score]
    objectness_threshold:
    iou_threshold: 
    '''

    dets = dets[dets[:, 4] > objectness_threshold]
    if len(dets) == 0: return {}

    pred_bboxes = dets[:, 0: 4]
    pred_classes = np.argmax(dets[:, 5:], axis=1)
    pred_scores = np.max(dets[:, 5:], axis=1)

    pred_bboxes = pred_bboxes[pred_scores > class_threshold]
    pred_classes = pred_classes[pred_scores > class_threshold]
    pred_scores = pred_scores[pred_scores > class_threshold]

    if len(pred_scores) == 0: return {}

    nclass = len(dets[0, 5:])

    result = {}

    for i in range(nclass):

        _class_i_index = np.where(pred_classes == i)[0]
        
        if len(_class_i_index) == 0:
            continue
        
        # print('_class_i_index: ', _class_i_index)
        # print('i: ', i)

        _dets_i_bboxes = pred_bboxes[_class_i_index]
        _dets_i_bboxes = ops_transform.xywh2xyxy(_dets_i_bboxes)
        _dets_i_scores = pred_scores[_class_i_index]
       
        keeps = nms_per_class(_dets_i_bboxes, _dets_i_scores, iou_threshold)
        _dets_i_bboxes = _dets_i_bboxes[keeps]
        _dets_i_scores = _dets_i_scores[keeps]

        # print(keeps)
        result[i] = [_dets_i_bboxes, _dets_i_scores]

    return result


def soft_nms():
    pass
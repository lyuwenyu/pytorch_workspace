from PIL import Image
import numpy as np 



def flip_lr(img, bbox=None):
    '''
    img: PIL
    bbox: numpy (x1 y1 x2 y2)
    '''
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    if bbox is not None:
        new_bbox = np.zeros_like(bbox)
        new_bbox[:, 0] = img.size[0] - bbox[:, 0] - (bbox[:, 2] - bbox[:, 0])
        new_bbox[:, 2] = img.size[0] - bbox[:, 2] + (bbox[:, 2] - bbox[:, 0])
        new_bbox[:, 1] = bbox[:, 1]
        new_bbox[:, 3] = bbox[:, 3]
        
        return img, new_bbox
    
    return img



def flip_tb(img, bbox=None):
    '''
    img: PIL
    bbox: numpy (x1 y1 x2 y2)
    '''
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    
    if bbox is not None:
        new_bbox = np.zeros_like(bbox)
        new_bbox[:, 1] = img.size[1] - bbox[:, 1] - (bbox[:, 3] - bbox[:, 1])
        new_bbox[:, 3] = img.size[1] - bbox[:, 3] + (bbox[:, 3] - bbox[:, 1])
        new_bbox[:, 0] = bbox[:, 0]
        new_bbox[:, 2] = bbox[:, 2]

        return img, new_bbox
    
    return img


def xyxy2xywh(bboxes, size):
    ''' x1 y1 x2 y2 -> cx cy w h '''
    assert isinstance(bboxes, np.ndarray), 'boxes should be ndarray.'

    new_bboxes = np.zeros_like(bboxes) # bboxes.copy() # copy.deepcopy(bboxes)
    new_bboxes[:, 0] = (bboxes[:, 0] + bboxes[:, 2]) / 2 / size[0]
    new_bboxes[:, 1] = (bboxes[:, 1] + bboxes[:, 3]) / 2 / size[1]
    new_bboxes[:, 2] = (bboxes[:, 2] - bboxes[:, 0]) / size[0]
    new_bboxes[:, 3] = (bboxes[:, 3] - bboxes[:, 1]) / size[1]

    return new_bboxes


def xyhw2xyxy(bboxes, size):
    '''
    '''
    new_bboxes = np.zeros_like(bboxes)
    new_bboxes[:, 0] = (bboxes[:, 0] - bboxes[:, 2] / 2) * size[0]
    new_bboxes[:, 1] = (bboxes[:, 1] - bboxes[:, 3] / 2) * size[1]
    new_bboxes[:, 2] = (bboxes[:, 0] + bboxes[:, 2] / 2) * size[0]
    new_bboxes[:, 3] = (bboxes[:, 1] + bboxes[:, 3] / 2) * size[1]

    return new_bboxes

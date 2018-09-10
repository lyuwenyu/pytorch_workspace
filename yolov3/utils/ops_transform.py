from PIL import Image
import numpy as np 



def flip_lr(img, bbox=None):
    '''
    img: PIL
    bbox: numpy (x1 y1 x2 y2)
    '''
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    if bbox is not None:
        bbox[:, 0] = img.size[0] - bbox[:, 0]
        bbox[:, 2] = img.size[0] - bbox[:, 2]
        return img, bbox
    
    return img



def flip_tb(img, bbox=None):
    '''
    img: PIL
    bbox: numpy (x1 y1 x2 y2)
    '''
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    
    if bbox is not None:
        bbox[:, 1] = img.size[1] - bbox[:, 1]
        bbox[:, 3] = img.size[1] - bbox[:, 3]
        return img, bbox
    
    return img



def xyxy2xywh(bboxes):
    ''' x1 y1 x2 y2 -> cx cy w h '''
    assert isinstance(bboxes, np.ndarray), 'boxes should be ndarray.'

    new_bboxes = np.zeros_like(bboxes) # bboxes.copy() # copy.deepcopy(bboxes)
    new_bboxes[: 0] = (bboxes[: 0] + bboxes[: 2]) / 2
    new_bboxes[: 1] = (bboxes[: 1] + bboxes[: 3]) / 2
    new_bboxes[: 2] = bboxes[: 2] - bboxes[: 0]
    new_bboxes[: 3] = bboxes[: 3] - bboxes[: 1]

    return new_bboxes

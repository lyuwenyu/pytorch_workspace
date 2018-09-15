from PIL import Image
import numpy as np
from utils.ops_show_bbox import show_bbox
from utils.ops_perpective import perspective_operation


def xyxy2xywh(bboxes, size=None):
    ''' x1 y1 x2 y2 -> cx cy w h '''
    assert isinstance(bboxes, np.ndarray), 'boxes should be ndarray.'

    new_bboxes = np.zeros_like(bboxes)  # bboxes.copy() # copy.deepcopy(bboxes)
    new_bboxes[:, 0] = (bboxes[:, 0] + bboxes[:, 2]) / 2 
    new_bboxes[:, 1] = (bboxes[:, 1] + bboxes[:, 3]) / 2
    new_bboxes[:, 2] = (bboxes[:, 2] - bboxes[:, 0])
    new_bboxes[:, 3] = (bboxes[:, 3] - bboxes[:, 1])

    if size is not None: # nomalize
        new_bboxes[:, [0, 2]] /= size[:, 0]
        new_bboxes[:, [1, 3]] /= size[:, 1]

    return new_bboxes


def xywh2xyxy(bboxes, size=None):
    '''cx cy w h -> x1 y1 x2 y2'''
    assert isinstance(bboxes, np.ndarray), 'boxes should be ndarray.'

    new_bboxes = np.zeros_like(bboxes)
    new_bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2
    new_bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] / 2
    new_bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2] / 2
    new_bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3] / 2

    if size is not None: # reverse normalize
        new_bboxes[:, [0, 2]] *= size[0]
        new_bboxes[:, [1, 3]] *= size[1]

    return new_bboxes


def flip_lr(img, bbox=None):
    '''
    img: PIL
    bbox: numpy (x1 y1 x2 y2)
    '''
    img = img.transpose(Image.FLIP_LEFT_RIGHT)

    if bbox is not None:
        bbox = np.array(bbox)
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
        bbox = np.array(bbox)
        new_bbox = np.zeros_like(bbox)
        new_bbox[:, 1] = img.size[1] - bbox[:, 1] - (bbox[:, 3] - bbox[:, 1])
        new_bbox[:, 3] = img.size[1] - bbox[:, 3] + (bbox[:, 3] - bbox[:, 1])
        new_bbox[:, 0] = bbox[:, 0]
        new_bbox[:, 2] = bbox[:, 2]

        return img, new_bbox

    return img


def random_perspective(image, bbox, label, skew_type='RANDOM'):
    '''
    image: PIL.Image
    bbox: box
    label: box label
    make sure bbox have its corresponding label, and one to one
    '''
    _img, M, _ = perspective_operation([image], magnitude=0.8, skew_type=skew_type)
    bbx = np.array(bbox)
    points = np.vstack([bbx[:, :2], bbx[:, 2:]])
    points = np.hstack([points, np.ones((len(points), 1), dtype=np.float32)])
    points = np.dot(points, M.T)
    points[:, 0] = points[:, 0] / (points[:, -1] + 1e-10)
    points[:, 1] = points[:, 1] / (points[:, -1] + 1e-10)
    points = np.hstack([points[:int(len(points)/2), :2], points[int(len(points)/2):, :2]])
    
    points[:, 0] = np.minimum(np.maximum(0, points[:, 0]), _img.size[0] - 1)
    points[:, 1] = np.minimum(np.maximum(0, points[:, 1]), _img.size[1] - 1)
    points[:, 2] = np.minimum(np.maximum(0, points[:, 2]), _img.size[0] - 1)
    points[:, 3] = np.minimum(np.maximum(0, points[:, 3]), _img.size[1] - 1)
    
    areas = (points[:, 3] - points[:, 1]) * (points[:, 2] - points[:, 0])
    index = np.where(areas > 200)[0]
    if len(index) == 0:
        print('no objects...')
        # return None, None
        return image, bbox, label
        
    label = [label[i] for i in index]
    points = points[index]

    assert len(label) == len(points)

    return _img, points, label


def pad_resize(img, bbox=None, size=None):
    '''
    pad and resize image and corresponding bbox

    img: PIL 
    bbox: numpy
    '''

    w, h = img.size

    if size is None:
        size = [max(w, h)] * 2

    scale = min(size[0] / w, size[1] / h)
    nw, nh = int(scale * w), int(scale * h)
    img = img.resize((nw, nh), Image.BICUBIC)

    pad = (size[0] - nw) // 2, (size[1] - nh) // 2
    new_img = Image.new('RGB', size=size, color=(128, 128, 128))
    new_img.paste(img, pad)

    if bbox is not None:
        bbox = np.array(bbox)
        bbox[:, 0] = np.maximum(bbox[:, 0] * scale + pad[0], 0)
        bbox[:, 1] = np.maximum(bbox[:, 1] * scale + pad[1], 0)
        bbox[:, 2] = np.minimum(bbox[:, 2] * scale + pad[0], size[0] - 1)
        bbox[:, 3] = np.minimum(bbox[:, 3] * scale + pad[1], size[1] - 1)
        # show_bbox(new_img, bbox, new_img.size)

        return new_img, bbox
    
    return new_img


def resize(img, bbox=None, size=None):
    '''
    resize
    '''
    w, h = img.size
    scale_w, scale_h = size[0] / w, size[1] / h

    img = img.resize(size)

    if bbox is not None:
        bbox = np.array(bbox)
        bbox[:, 0] = bbox[:, 0] * scale_w
        bbox[:, 1] = bbox[:, 1] * scale_h
        bbox[:, 2] = bbox[:, 2] * scale_w
        bbox[:, 3] = bbox[:, 3] * scale_h

        return img, bbox

    return img


def bbox_iou(boxa, boxb):
    '''comput iou bettwen boxes'''
    pass


def random_crop(img, bbox, label):
    '''crop
    '''
    pass
    
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import torchvision.transforms as transforms

def show_bbox(img, bbox=None, names=None, xyxy=True, normalized=False, show=True):
    '''
    '''
    draw = ImageDraw.Draw(img)
    bbox = np.array(bbox)

    if xyxy:
        # if normalized:
        #     bbox[:, [0, 2]] *= img.size[0]
        #     bbox[:, [1, 3]] *= img.size[1]
        new_bbox = bbox

    else:
        new_bbox = np.zeros_like(bbox)
        new_bbox[:, 0] = bbox[:, 0] - bbox[:, 2] / 2
        new_bbox[:, 1] = bbox[:, 1] - bbox[:, 3] / 2
        new_bbox[:, 2] = bbox[:, 0] + bbox[:, 2] / 2
        new_bbox[:, 3] = bbox[:, 1] + bbox[:, 3] / 2

    if normalized:
        new_bbox[:, [0, 2]] *= img.size[0]
        new_bbox[:, [1, 3]] *= img.size[1]
    
    for ii, bb in enumerate(new_bbox):
        draw.rectangle(tuple(bb), outline='red')
        if names is not None:
            draw.text((bb[0], bb[1]), names[ii], )
            
    if show:
        img.show()

    return img


def show_tensor_bbox(image_tensor, bbox_tensor, xyxy=False, normalized=True):
    ''' for torch dataset.
    image: c, h, w (totensor)
    bboxes: n 6, [mask-0/1, label-0/cls, bbox-4/x/y/h/w]
    '''
    topil = transforms.ToPILImage()

    image = topil(image_tensor.cpu().data)
    bbox_tensor = bbox_tensor[bbox_tensor[:, 0] == 1] # filter mask=0, no object
    bbox_tensor = bbox_tensor.cpu().data.numpy()

    show_bbox(image, bbox_tensor[:, 2: ], names=bbox_tensor[:, 1:2], xyxy=xyxy, normalized=normalized)


def show_preds_bbox(image, preds):
    pass


def show_polygon(img, bbox=None, names=None, xyxy=True, normalized=False, show=True):
    '''
    '''
    draw = ImageDraw.Draw(img)
    bbox = np.array(bbox)

    if xyxy:
        new_bbox = bbox
    else:
        # TODO
        pass

    if normalized:
        new_bbox[:, [0, 2, 4, 6]] *= img.size[0]
        new_bbox[:, [1, 3, 5, 7]] *= img.size[1]
    
    for bb in new_bbox:
        draw.polygon(tuple(bb), outline='red')

    if show:
        img.show()
    
    return img
    
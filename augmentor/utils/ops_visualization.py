from PIL import Image, ImageDraw
import numpy as np


def show_points(img, points, normalized=False, radius=3, color='red', show=True):
    '''
    img: PIL Image
    points: [(x, y), (x, y), ...]
    '''
    draw = ImageDraw.Draw(img)

    if normalized:
        points = np.array(points)
        points[:, 0] *= img.size[0]
        points[:, 1] *= img.size[1]
        
    for x, y in points:
        rect = (x-radius, y-radius, x+radius, y+radius)
        draw.ellipse(rect, fill=color)
    
    if show:
        img.show()

    return img




def show_bbox(img, bbox=None, names=None, xyxy=True, normalized=False, show=True):
    '''
    '''
    draw = ImageDraw.Draw(img)
    bbox = np.array(bbox)

    if xyxy:
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
    
    for bb in new_bbox:
        draw.rectangle(tuple(bb), outline='red')

    if show:
        img.show()

    return img
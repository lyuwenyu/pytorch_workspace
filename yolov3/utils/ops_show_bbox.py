import numpy as np
from PIL import Image
from PIL import ImageDraw


def show_bbox(img, bbox=None, names=None, xyxy=True, normalize=True):

    draw = ImageDraw.Draw(img)
    bbox = np.array(bbox)
    if xyxy:
        
        if normalize:
            bbox[:, [0, 2]] *= img.size[0]
            bbox[:, [1, 3]] *= img.size[1]
    
        new_bbox = bbox

    else:

        new_bbox = np.zeros_like(bbox)
        new_bbox[:, 0] = bbox[:, 0] - bbox[:, 2] / 2
        new_bbox[:, 1] = bbox[:, 1] - bbox[:, 3] / 2
        new_bbox[:, 2] = bbox[:, 0] + bbox[:, 2] / 2
        new_bbox[:, 3] = bbox[:, 1] + bbox[:, 3] / 2

        if normalize:
            new_bbox[:, [0, 2]] *= img.size[0]
            new_bbox[:, [1, 3]] *= img.size[1]
            # new_bbox[:, 2] *= img.size[0]
            # new_bbox[:, 3] *= img.size[1]
    
    for bb in new_bbox:
        draw.rectangle(tuple(bb), outline='red')

    img.show()

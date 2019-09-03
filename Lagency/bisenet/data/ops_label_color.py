from collections import OrderedDict
from PIL import Image, ImageDraw
import numpy as np 

# name, r, g, b
# backgroud 0 0 0
label_color = OrderedDict([
    ['background', (0, 0, 0)],
    ['parking_spot', (128, 0, 0)],
    ['car', (0, 128, 0)],
    ['line', (0, 0, 128)],
    ['pillar', (128, 128, 0)],
])

label_map = dict(zip(label_color.keys(), range(len(label_color.keys()))))
label_color_map = {label_color[k]: v for k, v in label_map.items()}
label_id_color = {label_map[k]: v for k, v in label_color.items()}

# print(label_id_color)
# print(label_color)
# print(label_map)
# print(label_color_map)


def label2png(size, labels, points):
    '''
    labels: list
    points: list
    return: Image
    '''
    anno = Image.new('RGB', size, color=label_color['background'])
    anno_draw = ImageDraw.Draw(anno)

    for lab, pts in zip(labels, points):
        # print(lab, pts)
        anno_draw.polygon(pts, fill=label_color[lab])
    
    return anno


def png2label(size, png):
    '''
    '''
    anno = Image.open(png)
    ann_arr = np.array(anno)
    ann_msk = np.zeros(ann_arr.shape[:-1])

    for k, v in label_color_map.items():
        msk = np.all(ann_arr == np.array(k).reshape(1, 1, 3), axis=2)
        ann_msk[msk] = v
    
    return ann_msk
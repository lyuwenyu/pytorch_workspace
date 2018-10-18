import json
import numpy as np 
import glob

from PIL import Image
from PIL import ImageDraw

from labelme import utils

import cv2
import albumentations

def load_json(path):
    '''for labelme 
    '''

    def flat(lists):
        result = []
        for listx in lists:
            result += listx
        return result

    with open(path, 'rb') as f:
        data = json.load(f)
    
    img = utils.img_b64_to_arr(data['imageData'])
    # mask, labels = utils.labelme_shapes_to_label(img.shape, data['shapes'])
    # print(mask.shape, labels)
    # masks = [(mask == i).astype(np.uint8) for i in range(len(labels))]
    # result = utils.draw_label(mask, img, labels, colormap=None)
    # Image.fromarray(result).show()

    blob = {
        'imagePath': data['imagePath'],
        'imageData': Image.fromarray(img),
        'labels': [],
        'points': []
    }
    
    for info in data['shapes']:
        blob['labels'] += [info['label']]
        blob['points'] += [flat(info['points'])]

    assert len(blob['labels']) == len(blob['points']), ''

    return blob


if __name__ == '__main__':

    paths = glob.glob('/home/wenyu/workspace/dataset/fisheye/quadrilateral/*.json')
    for path in paths:
        data = load_json(path)
    
    img = Image.open('/home/wenyu/workspace/dataset/fisheye/quadrilateral/153795440566.jpg')
    draw = ImageDraw.Draw(img)

    for p in data['points']:
        draw.polygon(tuple(p), outline='red')

    img.show()
    # print(data['points'])
import json
import os
from labelme import utils
from PIL import Image
import numpy as np

def labelme_parser(path):
    '''for labelme
    '''
    with open(path, 'rb') as f:
        data = json.load(f)

    image = utils.img_b64_to_arr(data['imageData'])
    blob = {
        'imagePath': data['imagePath'],
        'imageData': image,
        'labels': [],
        'points': [],
        'height': image.shape[0],
        'width': image.shape[1],
    }

    for info in data['shapes']:
        blob['labels'] += [info['label']]
        blob['points'] += [info['points']]

    blob['labels'] = np.array(blob['labels'])
    blob['points'] = np.array(blob['points'])

    xmin = np.maximum(np.min(blob['points'][:, :, 0], axis=1)[:, np.newaxis], 0)
    xmax = np.minimum(np.max(blob['points'][:, :, 0], axis=1)[:, np.newaxis], image.shape[1] - 1)
    ymin = np.maximum(np.min(blob['points'][:, :, 1], axis=1)[:, np.newaxis], 0)
    ymax = np.minimum(np.max(blob['points'][:, :, 1], axis=1)[:, np.newaxis], image.shape[0] - 1)
    bboxes = np.hstack([xmin, ymin, xmax, ymax])
    blob['bboxes'] = bboxes

    assert len(blob['labels']) == len(blob['points']), ''

    return blob


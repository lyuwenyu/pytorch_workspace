import numpy as np
import json
from PIL import Image, ImageDraw
from utils.ops_perspective import perspective_operation
import os
import random

from multiprocessing import Pool
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_json_file', default='', nargs='+')
parser.add_argument('--output_json_file', default='')
parser.add_argument('--img_out_dir', default='')
parser.add_argument('--n_aug_per', default=5, type=int)
args = parser.parse_args()

data_augm_dir = args.img_out_dir

def func(data):

    try:
        img = Image.open(data['filename'])
    except:
        print(data['filename'])
        return -1

    augment_blobs = []

    for i in range(args.n_aug_per):

        blob = {'filename': '',
                'names': [], 
                'bboxes': [],
                'height': 0,
                'width': 0
        }

        _img, M, _ = perspective_operation([img], magnitude=0.8, skew_type='RANDOM')
        bbx = np.array(data['bboxes'])
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
            continue

        ll = os.path.join(data_augm_dir, '{}_{:0>2}.jpg'.format(os.path.basename(data['filename'])[:-4], i))
        _img.save(ll)

        blob['filename'] = ll
        blob['names'] = [data['names'][i] for i in index] 
        blob['height'] = _img.size[1]
        blob['width'] = _img.size[0]
        blob['bboxes'] = [tuple(lin) for lin in points[index]]

        assert len(blob['names']) == len(blob['bboxes']), 'wrong in augmentation bbox and names'

        # draw = ImageDraw.Draw(_img)
        # for pt in blob['bboxes']:
        #     draw.rectangle( tuple(pt), outline='red')
        # _img.show()
        # break

        augment_blobs += [blob]

    return augment_blobs


if __name__ == '__main__':

    raw_data = {
        'classes': [],
        'raw_data': []
    }
    for jsonfile in args.input_json_file:
        with open(jsonfile, 'r') as f:
            _raw_data = json.load(f)
            raw_data['classes'] += _raw_data['classes']
            raw_data['raw_data'] += _raw_data['raw_data']

    print(raw_data['classes'])
    print(len(raw_data['raw_data']))

    random.shuffle(raw_data['raw_data'])

    pool = Pool(16)
    
    res = pool.map(func, raw_data['raw_data'])
    pool.close()
    pool.join()

    data = []
    info = dict(zip(raw_data['classes'], [0] * len(raw_data['classes'])))

    for d in res:
        data += d

    for d in data:
        for n in d['names']:
            info[n] += 1
        
    print('augmentation images: ', len(data))
    print('data info: ', info)

    _data = {'raw_data': data, 'classes': raw_data['classes']}

    with open(args.output_json_file, 'w') as f:
        json.dump(_data, f)



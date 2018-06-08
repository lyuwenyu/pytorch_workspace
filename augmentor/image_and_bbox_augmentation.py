from PIL import Image
import Augmentor
from xml.etree import cElementTree as ET 
import numpy as np 
import random
import glob
import os
import json
from collections import OrderedDict
from multiprocessing import Pool 


p = Augmentor.Pipeline()
p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
p.random_distortion(probability=1.0, grid_height=50, grid_width=50, magnitude=2)
# p.skew(probability=1)


def new_image(bbx, h, w):
    _tmp = np.zeros(shape=(h, w))
    _tmp[bbx[1]: bbx[3], bbx[0]: bbx[2]] = 255
    img_tmp = Image.fromarray(_tmp)
    return img_tmp

def sample(p, images):
    for operation in p.operations:
        r = round(random.uniform(0, 1), 1)
        if r <= operation.probability:
            images = operation.perform_operation(images)
    return images


def generate(xml_path='./1.xml', img_path='./1.jpg'):
    
    img = Image.open(img_path)
    w, h = img.size

    with open(xml_path, 'r') as f:
        raw_data = ET.fromstring(f.read())
        
        bbox, name = [], []
        for c in raw_data.getchildren():
            if c.tag != 'object':
                continue
            
            name += [c.find('name').text]

            bbx = c.find('bndbox')
            bbx = [int(i.text) for i in bbx.getchildren()]
            bbox += [bbx]
            
        images = [new_image(bb, h, w) for bb in bbox]

        images = sample(p, [img]+images)
        new_bbox = [im.getbbox() for im in images[1:]]

        blob = [ (n, b) for n, b in zip(name, new_bbox) if b is not None ]
        print(blob)

    return images[0], blob


def func(opts): #outdir='./outdir', num=100):

    if not os.path.exists(opts['outdir']):
        os.makedirs(opts['outdir'])

    for i in range(opts['num']):
        
        new_img, new_info = generate(opts['xml'], opts['image'])

        new_img.save(os.path.join(opts['outdir'], '{}_{:0>5}.jpg'.format(os.path.basename(opts['xml'])[:-4], i)))

        with open(os.path.join(opts['outdir'], '{}_{:0>5}.json'.format(os.path.basename(opts['xml'])[:-4], i)), 'w') as fx:
            json.dump(new_info, fx)



if __name__ == '__main__':

    dirname='./'
    outdir = './outdir'
    num_per_img = 100

    xmls = glob.glob(os.path.join(dirname, '*.xml'))
    imgs = [x[:-3]+'jpg' for x in xmls]

    opts = [{'xml': xm, 'image': im, 'outdir': outdir, 'num': num_per_img} for xm, im in zip(xmls, imgs)]

    pool = Pool(10)
    pool.map(func, opts)
    pool.close()
    pool.join()



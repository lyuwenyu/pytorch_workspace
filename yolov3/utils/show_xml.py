import os
from PIL import Image
from PIL import ImageDraw

import numpy as np

from utils.ops_parse_xml import parse_xml
from utils.ops_pad_resize import pad_resize
from utils.ops_transform import flip_lr, flip_tb

root = '/home/wenyu/workspace/dataset/fisheye/dataset_zc/20180815_record5_KeJiYuan/'
ann = os.path.join(root, 'anno', '20180815_record5_camera_2_00004981.xml')
path = ann.replace('anno', 'img').replace('xml', 'jpg')
print(path)

img = Image.open(path)
draw = ImageDraw.Draw(img)

blob = parse_xml(ann)


for bbx in blob['bboxes']:

    draw.rectangle(tuple(bbx), outline='red')

img.show()


# img = Image.open('./_model/dog-cycle-car.png')

img = Image.open(path)

print( np.array(blob['bboxes']).shape)

img, bboxes = pad_resize(img, np.array(blob['bboxes']), size=(512, 512))
draw = ImageDraw.Draw(img)

for bbx in bboxes:
    draw.rectangle(tuple(bbx), outline='red')
img.show()



img = Image.open(path)
img, bboxes = pad_resize(img, np.array(blob['bboxes']), size=(512, 512))
img, bboxes = flip_lr(img, bboxes)

draw = ImageDraw.Draw(img)

for bbx in bboxes:
    draw.rectangle(tuple(bbx), outline='red')
img.show()



img = Image.open(path)
img, bboxes = pad_resize(img, np.array(blob['bboxes']), size=(512, 512))
img, bboxes = flip_lr(img, bboxes)
img, bboxes = flip_tb(img, bboxes)

draw = ImageDraw.Draw(img)

for bbx in bboxes:
    draw.rectangle(tuple(bbx), outline='red')
img.show()
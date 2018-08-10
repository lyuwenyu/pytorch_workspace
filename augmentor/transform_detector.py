# Stupid python path shit.
# Instead just add darknet.py to somewhere in your python path
# OK actually that might not be a great idea, idk, work in progress
# Use at your own risk. or don't, i don't care
import time
import sys, os
sys.path.append(os.path.join(os.getcwd(),'python/'))

import darknet as dn
import pdb
from PIL import Image
from PIL import ImageDraw
from ops import perspective_operation, NMS
import numpy as np


#dn.set_gpu(0)
#dn.set_cpu()
net = dn.load_net("cfg/yolov3.cfg", "yolov3.weights", 0)
meta = dn.load_meta("cfg/coco.data")

#r = dn.detect(net, meta, "data/bedroom.jpg")
#print r

# And then down here you could detect a lot more images like:
#r = dn.detect(net, meta, "data/eagle.jpg")
#print r
#r = dn.detect(net, meta, "data/giraffe.jpg")
#print r
#r = dn.detect(net, meta, "data/horses.jpg")
#print r


# start = time.time()
# r = dn.detect(net, meta, "data/dog.jpg")
# first = time.time()
# print('the first image. {:.5}'.format(first-start))


# r = dn.detect(net, meta, "data/person.jpg")
# second = time.time()

# print('second image. {:.5}'.format(second-first))


# r = dn.detect(net, meta, "data/dog.jpg")
# print(time.time()-second)


name = 'data/003.jpg'

img_origin = Image.open(name)
# img_origin.save('output/img.jpg')
drawobj = ImageDraw.Draw(img_origin)
W, H = img_origin.size

lines = []


r = dn.detect(net, meta, name)
for _r in r:
    
    _cx, _cy, _w, _h = _r[2]

    drawobj.rectangle((_cx-_w/2, _cy-_h/2, _cx+_w/2, _cy+_h/2), outline='blue')
    drawobj.text((_cx-_w/2, _cy-_h/2), _r[0], fill='blue')

    lines += [ (_r[0], _r[1], _cx-_w/2, _cy-_h/2, _cx+_w/2, _cy+_h/2), ]
    
# img.show()
print('---------', r)
img_origin.save('output/origin.jpg')
# img_origin.show()


tic = time.time()

img = Image.open(name)

n = 10
for i in range(n):

    imgs, M, M_reverse = perspective_operation([img], magnitude=1.0, skew_type='TILT')
    imgs[0].save('test.jpg')

    r = dn.detect(net, meta, 'test.jpg')

    draw = ImageDraw.Draw(imgs[0])

    for _r in r:

        _cx, _cy, _w, _h = _r[2]
        # draw.rectangle((_cx-_w/2, _cy-_h/2, _cx+_w/2, _cy+_h/2), outline='blue')
        # draw.text((_cx-_w/2, _cy-_h/2), _r[0], fill='blue')
        
        points = np.array([[_cx-_w/2, _cy-_h/2, 1], [_cx+_w/2, _cy+_h/2, 1], [_cx-_w/2, _cy+_h/2, 1], [_cx+_w/2, _cy-_h/2, 1]])
        points = np.dot(points, M_reverse.T) # np.dot(M_reverse, points.T).T
        points[:, 0] = points[:, 0] / (points[:, -1] + 1e-10)
        points[:, 1] = points[:, 1] / (points[:, -1] + 1e-10)

        # drawobj.polygon([(points[0, 0], points[0, 1]), (points[2, 0], points[2, 1]),(points[1, 0], points[1, 1]), (points[3, 0], points[3, 1])], outline='yellow')
        # points[:, 0] = 

        # new_pos1 = np.dot(M_reverse, np.array([_cx-_w/2, _cy-_h/2, 1]))
        # new_pos1[0] = new_pos1[0] / new_pos1[2]
        # new_pos1[1] = new_pos1[1] / new_pos1[2]
        # new_pos2 = np.dot(M_reverse, np.array([_cx+_w/2, _cy+_h/2, 1]))
        # new_pos2[0] = new_pos2[0] / new_pos2[2]
        # new_pos2[1] = new_pos2[1] / new_pos2[2]
        # print( new_pos1[0], new_pos1[1], new_pos2[0], new_pos2[1], type(new_pos1[0])
        # drawobj.rectangle((new_pos1[0], new_pos1[1], new_pos2[0], new_pos2[1]), outline='blue')
        # lines += [ (_r[0], _r[1], new_pos1[0], new_pos1[1], new_pos2[0], new_pos2[1]), ]
        if points[0, 0] > 0 and points[0, 1] > 0 and points[1, 0] < W and  points[1, 1] < H:
            lines += [ (_r[0], _r[1], points[0, 0], points[0, 1], points[1, 0], points[1, 1]), ]


    # imgs[0].show()
    imgs[0].save('output/{:0>5}.jpg'.format(i))


res = dict()
classes = set([l[0] for l in lines])
for c in classes:

    dets = [ list(map(np.float32, l[1:])) for l in lines if l[0] == c]
    dets = np.array(dets)
    dets = np.hstack((np.zeros((dets.shape[0], 1)), dets))

    keeps = NMS(dets, threshold=0.5)

    print('----------', c, keeps)

    dets = dets[keeps]
    res[c] = dets[:, 1:]
    # print(dets)


print('---------')
print(res)

for c in res:

    for bbx in res[c]:
        drawobj.rectangle((bbx[1], bbx[2], bbx[3], bbx[4]), outline='red')
        drawobj.text((bbx[1], bbx[2]), c, fill='red')


print('time: ',time.time() - tic)

with open('./output/result.txt', 'w') as f:
    for lin in lines:
        _lin = '\t'.join(list(map(str, lin)))
        _lin += '\n'
        f.write(_lin)


img_origin.save('output/final.jpg')
img_origin.show()
import numpy as np
import time
from PIL import Image, ImageDraw

import torch
from torchvision import transforms

from model.build_model import DarkNet
from utils.ops_nms import NMS
from utils.ops_show_bbox import show_bbox
from utils import ops_transform

import argparse
import glob
import random

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='./_model/dog-cycle-car.png')
parser.add_argument('--dim', type=int, default=(416, 416)) # h, w
parser.add_argument('--num_classes', type=int, default=10)
args = parser.parse_args()

def pre_image(img, inp_dim):
    totensor = transforms.ToTensor()
    # w, h = img.size
    # _img = img.crop((0, int(h*2.0//7), w-1, h-1-(h*1.0//7)))
    _img = ops_transform.resize(img, size=(inp_dim[1], inp_dim[0]))
    return totensor(_img).unsqueeze(0)

dim = args.dim
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

model = DarkNet('./_model/yolov3_v1.cfg', cls_num=args.num_classes)
model.load_state_dict(torch.load('./output/state-ckpt-epoch-00200', map_location='cpu')['model'])

model.eval()
model = model.to(device=device)

def save_dets(blobs, classes, out_dir=''):
    '''
    key: file name
    value: bs, cs, ss
    '''
    for clss in classes:
        f = open('/home/wenyu/workspace/detection/result/{}.txt'.format(clss), 'w')
        for k in blobs:
            bs, cs, ss = blobs[k]
            for i in range(len(cs)):
                if cs[i] == clss:
                    ll = '{} {} {} {} {} {}\n'.format(k,
                                                    ss[i], 
                                                    bs[i][0],
                                                    bs[i][1], 
                                                    bs[i][2], 
                                                    bs[i][3])
                    f.write(ll)
        f.close()


def show_detection(img, bs, cs, ss, save_dir=''):
    '''
    img: PIL image
    bs: bounding boxs
    cs: classes
    ss: scores
    '''
    draw = ImageDraw.Draw(img)

    for i in range(len(bs)):
        _b = tuple(bs[i])
        draw.rectangle(_b, outline='blue', )
        # draw.text((_b[0], _b[1]), text=cs[i]+' '+str(ss[i]), fill='blue',)

    if len(save_dir) > 0:
        # img.save(save_dir+f'/{time.time()}.jpg')
        img.save(save_dir+'/{}.jpg'.format(time.time()))
    else:
        img.show()



if __name__ == '__main__':

    import os
    
    with open('./data/fisheye.names', 'r') as f:
        names = [l.strip() for l in f.readlines()]
        
    label_map = dict(zip(range(len(names)), names))

    blobs = dict()

    # paths = [args.path]

    random.shuffle(paths)
    paths = paths[:10]

    for i, path in enumerate(paths):

        lin = os.path.basename(path)[:-4]

        img = Image.open(path)
        w, h = img.size
        # img = img.crop((0, int(h*2.0//7), w-1, h-1-(h*1.0//7)))
        # w, h = img.size
        
        scale_w = 1.0 * w / dim[1]
        scale_h = 1.0 * h / dim[0]

        data = pre_image(img, dim).to(dtype=torch.float32, device=device)
        data = data.to(device=device)
        print('shape: ', data.size())

        tic = time.time()
        pred = model(data)[0].cpu().data.numpy()
        pred[:, [0, 2]] *= scale_w
        pred[:, [1, 3]] *= scale_h

        print('model', time.time() - tic)
        result = NMS(pred, objectness_threshold=0.9, class_threshold=0.3, iou_threshold=0.3)
        print('model nms', time.time() - tic)
        print('----------------------')

        bs, ss, cs = [], [], []
        for k, v in result.items():
            bs += [v[0]]
            ss += [v[1]]
            cs += [label_map[k]] * len(v[0])
        
        if len(bs) > 0:
            bs = np.concatenate(bs, axis=0)
            ss = np.concatenate(ss, axis=0)
        else:
            continue

        # print(bs, cs, ss)
        blobs[lin] = [bs, cs, ss]

        if i % 100 == 0:
            print(i, )

        # show_detection(img, bs, cs, ss, save_dir='/home/wenyu/workspace/detection/result/image')
        show_detection(img, bs, cs, ss,)

    # save_dets(blobs, names, '/home/wenyu/workspace/detection/result')
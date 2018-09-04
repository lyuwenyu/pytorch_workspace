import os
import torch
import torch.nn as nn
from collections import defaultdict

def parse_config(cfg):
    ''' parse yolov3 config '''

    result = []
    with open(cfg, 'r') as f:
        lines = f.readlines()
        lines = [lin.strip() for lin in lines if not lin.startswith('#')]
        lines = [lin for lin in lines if len(lin)]

    for lin in lines:
        
        if lin.startswith('['):
            result += [{}]
            result[-1]['type'] = lin[1:-1]
        else:
            k, v = lin.split('=')
            result[-1][k.strip()] =  v.strip()

    return result

if __name__ == '__main__':
    path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(os.path.dirname(path), '_model', 'yolov3.cfg')
    print(path)
    res = parse_config(path)
    print(len(res))
    # print(res)
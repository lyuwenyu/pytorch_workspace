import numpy as np


num_classes = 1
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
height = 640
width = 640
stride = 4

dataset = {
        'task': 'detection',
        'type': 'bbox',
        'debug': 0,
        'anno_format': 'labelme',
        'image_root': './',
        'image_ext': 'jpg',
        'anno_root': './',
        'anno_ext': 'json',

        'num_classes': num_classes,
        'class_names': [],
        'max_n': 50,
        'mean': mean,
        'std': std,
        
        'stride': stride,

        'augmentor': {'min_visibility': 0.25,
                      'min_area': 5, 
                      'bbox': True,
                      'smallest': 1024,
                      'random_crop': True,
                      'safe_crop': False, # keep all bbox
                      'min_max_height': (740, 1024),
                      'height': height,
                      'width': width,
                      'flip': True,
                     }
        }


network = {
        
        'heads': {'hm': num_classes, 'wh': 2, 'off': 2}
          
        }


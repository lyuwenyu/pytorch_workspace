import numpy as np
from PIL import Image
import albumentations
from albumentations import OneOf, Compose


def bbox_augmentation(cfg): 
    '''
    '''

    transforms = []
    
    if cfg['smallest'] > 0:
        transforms += [albumentations.SmallestMaxSize(max_size=cfg['smallest'], p=1.0)]
    
    if cfg.get('random_crop', 0):
        # transforms += [OneOf([albumentations.RandomCrop(height=1024, width=1024, p=0.8),
        #                     albumentations.RandomCrop(height=720, width=720, p=0.8),], p=1.),]
        if cfg.get('safe_crop', 0):
            transforms += [albumentations.RandomSizedBBoxSafeCrop(height=cfg['height'], width=cfg['width'], p=1.)] 
        else:
            transforms += [albumentations.RandomSizedCrop(cfg['min_max_height'], height=cfg['height'], width=cfg['width'], p=1.0)]

    if cfg.get('flip', 0):
        transforms += [albumentations.HorizontalFlip(p=0.5)]

    transforms += [albumentations.RandomBrightness(limit=0.2, p=0.3), 
            albumentations.RandomContrast(limit=0.2, p=0.3),
            albumentations.Blur(blur_limit=5, p=0.2),
            albumentations.GaussNoise(var_limit=(5, 20), p=0.2),
            albumentations.ChannelShuffle(p=0.2),]

    bbox_params = {'format': 'pascal_voc',
            'min_visibility': cfg['min_visibility'],
            'label_fields': ['labels'],
            'min_area': cfg['min_area']} if cfg['bbox'] else {}

    return Compose(transforms, bbox_params=bbox_params, p=1.)



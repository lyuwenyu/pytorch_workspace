import cv2
import numpy as np
from PIL import Image
import random
import os
import sys

import albumentations
from albumentations import Compose, OneOf

sys.path.insert(0, '../yolov3')
from yolov3.utils import ops_show_bbox

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

img = Image.open('dog-cycle-car.png')


def get_transforms(min_visibility=0.3, min_area=100, bbox=None):
    
    transforms = [
        albumentations.RandomCrop(height=400, width=400, p=1.),
        # OneOf([
        #     albumentations.Blur(),
        #     albumentations.Rotate(),
        # ], p=0.5),
        albumentations.RandomBrightness(p=0.5),
        albumentations.Resize(height=256, width=256, p=1.0)
    ]
    # random.shuffle(transforms)

    bbox_params = {'format': 'pascal_voc',
                    'min_visibility': min_visibility,
                    'label_fields': ['classes'],
                    'min_area': min_area
            } if bbox else {}

    return Compose(transforms, bbox_params=bbox_params, p=1)


transforms = get_transforms(bbox=True)

anns = {
    'image': np.array(img),
    'bboxes': [[99, 106, 445, 328], [103, 175, 241, 419], [39, 55, 86, 98], [367, 57, 532, 135]],
    'classes': [1, 2, 3, 4],
}
result = transforms(**anns)


image = Image.fromarray(result['image'])
bboxes = np.array(result['bboxes'])
ops_show_bbox.show_bbox(image, bboxes)



# segmentation

# img nparray / cv2
# aug = albumentations.RandomCrop(height=416, width=416, p=1.0)
# auged = aug(image=img, mask=mask)
# mask = auged['mask']
# img = auged['image']
# result = utils.draw_label(mask, img, labels, colormap=None)
# Image.fromarray(result).show()
# masks = [(mask == i).astype(np.uint8) for i in range(len(labels))]
# Image.fromarray(masks[0] * 100).show()

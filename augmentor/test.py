from multiprocessing import Pool
import random
import numpy as np
from PIL import Image
from skimage import transform, color
import skimage

import Augmentor
from Augmentor.Operations import Operation


def func(args):

    p = Augmentor.Pipeline('/home/wenyu/Desktop/test/')
    
    # p.set_seed(seed)
    p.rotate(probability=0.5, max_left_rotation=5, max_right_rotation=5)
    p.random_erasing(probability=0.5, rectangle_area=0.3)
    p.skew(probability=0.5)
    p.crop_random(probability=0.5, percentage_area=0.9)
    p.flip_left_right(probability=0.5)
    
    random.shuffle(p.operations)

    p.sample(args[0])



class NoiseResize(Operation):

    def __init__(self, probability):
        super(NoiseResize, self).__init__(probability)

    def perform_operation(self, images):

        def do(image):
            image = np.array(image)
            image = transform.rescale(image, scale=0.5)
            image = skimage.util.random_noise(image, mode='gaussian', clip=True)  
            image = transform.rescale(image, scale=2.0)
            image = Image.fromarray(skimage.util.img_as_ubyte(image))
            return image

        augmented_images = []
        for image in images:
            augmented_images.append(do(image))

        return augmented_images



if __name__ == '__main__':


    # seed = [1, 2, 3]
    # nums = [10, 10, 10]
    # pool = Pool(3)
    # pool.map(func, zip(nums, seed))
    # pool.close()
    # pool.join()

    p = Augmentor.Pipeline(source_directory='./train')
    p.add_operation(NoiseResize(probability=0.7))
    p.random_distortion(probability=0.5, grid_height=20, grid_width=20, magnitude=1)
    p.sample(10)

import Augmentor
from Augmentor import Pipeline
import random
from PIL import Image

class ImagesPipeline(Pipeline):
    '''
    Images: list, Image
    return: list, Image
    '''
    def transform(self, images, shuffle=False):
        '''perform operations on PIL image'''

        if not isinstance(images, list):
            images = [images]

        pilimgs = images

        if shuffle:
            random.shuffle(self.operations)

        for operation in self.operations:
            r = round(random.uniform(0, 1), 1)
            if r < operation.probability:
                pilimgs = operation.perform_operation(images)

        return pilimgs
import Augmentor
from Augmentor import Pipeline
import random
from PIL import Image

class ImagePipeline(Pipeline):

    def transform(self, image, shuffle=True):
        '''perform operations on PIL image'''

        pilimg = image

        if shuffle:
            random.shuffle(self.operations)

        for operation in self.operations:
            r = round(random.uniform(0, 1), 1)
            if r < operation.probability:
                pilimg = operation.perform_operation([image])[0]

        return pilimg
        


  

if __name__ == '__main__':

    pipeline = ImagePipeline()
    pipeline.random_brightness(probability=0.5, min_factor=0.5, max_factor=1.2)
    pipeline.random_distortion(probability=0.5, grid_height=20, grid_width=20, magnitude=1.0)
    pipeline.random_color(probability=0.5, min_factor=0.5, max_factor=1.5)
    pipeline.random_contrast(probability=0.5, min_factor=0.5, max_factor=1.5)

    image = Image.open('./_model/dog-cycle-car.png')

    newimage = pipeline.transform(image)
    newimage.show()
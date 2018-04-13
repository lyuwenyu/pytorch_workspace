from torchvision import transforms
import torchvision.transforms.functional as transformsF
import torch.utils.data as data 

import Augmentor
import glob 
from PIL import Image
import numpy as np 

import functools
import random

print(Augmentor.__version__)

class DatasetX(data.Dataset):


    def __init__(self, path='/home/wenyu/Desktop/test', use_augmentor=True, is_training=True):

        ## params
        self.use_augmentor = use_augmentor
        self.is_training = is_training

        ## paths
        self.lines = glob.glob(path + '/**/**/*.png')


        if use_augmentor:

            self.p = self._augmentor()
            # self.p.sample(50)

            # self.transforms = transforms.Compose([ self.p.torch_transform(), transforms.ToTensor() ])


        self.preprocess = transforms.Compose([  transforms.Resize(size=[224,224]),
                                                transforms.ToTensor(), 
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])


    def __len__(self):


        return len(self.lines)


    def __getitem__(self, i):
        
        
        img = Image.open(self.lines[i]).convert('RGB')

        imgs = [ img ]

        if self.is_training:
            
            if self.use_augmentor:
                
                _imgs = []
                _seed = random.randint(0, 100000)

                for x in imgs:

                    self.p.set_seed( _seed )
                    _imgs += [ Image.fromarray(self.p._execute_with_array(np.array(x))) ]

                imgs = _imgs

            else:

                imgs = self._transforms(imgs)


        imgs = [ self.preprocess(x) for x in imgs ]


        return imgs



    def _augmentor(self, path=None):

        p = Augmentor.Pipeline(path)

        p.crop_random(probability=0.5, percentage_area=0.8)
        p.resize(probability=1.0, width=512, height=512)
        p.flip_left_right(probability=0.5)
        p.rotate(probability=0.5, max_left_rotation=10, max_right_rotation=10)
        p.shear(probability=0.4, max_shear_left=10, max_shear_right=10)
        p.random_distortion(probability=0.3, grid_height=10, grid_width=10, magnitude=2)

        return p


    def _transforms(self, images=[]):
        
        if not isinstance(images, list):

            images = [images]

        params = transforms.RandomCrop.get_params(images[0], output_size=[224, 224]) ## get parameters
        images = [ transformsF.crop(im, *params) for im in images ]

        params = transforms.RandomRotation.get_params((-15, 15))
        images = [ transformsF.rotate(im, params) for im in images ]
        
        if random.random()> 0.5:
            images = [ transformsF.hflip(im) for im in images]


        return images


if __name__ == '__main__':


    dataset = DatasetX()

    
    dataset_loader = data.DataLoader(dataset, batch_size=10, num_workers=5) 

    for d in dataset_loader:

        print(len(d), d[0].size())


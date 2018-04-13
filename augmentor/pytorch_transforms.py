from torchvision import transforms
import torchvision.transforms.functional as transformsF
import torch.utils.data as data 
import torchvision.utils as vutils

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

        else:

            self.t = transforms.Compose([   transforms.RandomHorizontalFlip(),
                                            transforms.RandomRotation(10),
                                            transforms.RandomResizedCrop(256) ])



        self.preprocess = transforms.Compose([  transforms.Resize(size=[224,224]),
                                                transforms.ToTensor(), 
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])


    def __len__(self):


        return len(self.lines)


    def __getitem__(self, i):
        
        
        img = Image.open(self.lines[i]).convert('RGB')

        imgs = [ img, img ]  ## image in LIST will be transformed in the same way, e.g. image mask

        if self.is_training:
            

            _imgs = []
            _seed = random.randint(0, 10000000)

            if self.use_augmentor:
                
                for x in imgs:

                    # self.p.set_seed( _seed )
                    random.seed(_seed)
                    _imgs += [ Image.fromarray(self.p._execute_with_array(np.array(x))) ]

                

            else:

                # _imgs = self._transforms_func(imgs)

                ### NOT WORK WELL BELOW
                for x in imgs:
                    random.seed(_seed)
                    _imgs += [ self.t(x) ]


            imgs = _imgs


        imgs = [ self.preprocess(x) for x in imgs ]


        return imgs



    def _augmentor(self, path=None):

        p = Augmentor.Pipeline(path)

        p.crop_random(probability=0.5, percentage_area=0.8)
        p.resize(probability=1.0, width=512, height=512)
        p.flip_left_right(probability=0.5)
        p.rotate(probability=0.5, max_left_rotation=10, max_right_rotation=10)
        p.shear(probability=0.4, max_shear_left=10, max_shear_right=10)
        p.random_distortion(probability=0.3, grid_height=5, grid_width=5, magnitude=2)

        return p


    def _transforms_func(self, images=[]):
        
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


    dataset = DatasetX(use_augmentor=False)

    
    dataset_loader = data.DataLoader(dataset, batch_size=10, num_workers=5) 

    for i, d in enumerate(dataset_loader):

        print(len(d), d[0].size(), d[1].size())

        vutils.save_image(d[0], './d0_{}.jpg'.format(i), nrow=5, normalize=True)
        vutils.save_image(d[1], './d1_{}.jpg'.format(i), nrow=5, normalize=True)

from torchvision import transforms
import torchvision.transforms.functional as transformsF
import torch.utils.data as data 

import Augmentor
import glob 
from PIL import Image
import numpy as np 

import functools
import random



class DatasetX(data.Dataset):


    def __init__(self, path='/home/wenyu/Desktop/test', use_augmentor=True, is_training=True):

        ## params
        self.use_augmentor = use_augmentor
        self.is_training = is_training

        ## paths
        self.imgs = glob.glob(path + '/**/**/*.png')


        if use_augmentor:

            self.p = self._augmentor()
            # self.transforms = transforms.Compose([ self.p.torch_transform(), transforms.ToTensor() ])


        self.preprocess = transforms.Compose([ transforms.Resize(size=[224,224]),
                                                transforms.ToTensor(), 
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])


    def __len__(self):


        return len(self.imgs)


    def __getitem__(self, i):
        
        
        img = Image.open(self.imgs[i]).convert('RGB')

        imgs = [ img ]

        if self.is_training:
            
            if self.use_augmentor:
                
                imgs = [ Image.fromarray(self.p._execute_with_array(np.array(x))) for x in imgs ]

            else:

                imgs = self._transforms(imgs)


        imgs = [ self.preprocess(x) for x in imgs ]


        return imgs



    def _augmentor(self):

        p = Augmentor.Pipeline()

        p.rotate(probability=0.5, max_left_rotation=10, max_right_rotation=10)
        p.resize(probability=1.0, width=224, height=224)
        p.flip_left_right(probability=0.5)


        return p


    def _transforms(self, images=[]):
        
        if not isinstance(images, list):
            
            images = [images]

        params = transforms.RandomCrop.get_params(images[0], output_size=[224, 224]) ## get parameters
        images = [ transformsF.crop(im, *params) for im in images ]

        if random.random()> 0.5:
            images = [ transformsF.hflip(im) for im in images]


        return images


if __name__ == '__main__':


    dataset = DatasetX()

    
    dataset_loader = data.DataLoader(dataset, batch_size=10, num_workers=5) 

    for d in dataset_loader:

        print(len(d), d[0].size())


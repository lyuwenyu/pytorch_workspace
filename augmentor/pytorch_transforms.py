from torchvision import transforms
import torchvision.transforms.functional as transformsF
import torch.utils.data as data 

import Augmentor
import glob 
from PIL import Image
import numpy as np 

import functools

class DatasetX(data.Dataset):


    def __init__(self, path='/home/wenyu/Desktop/test', use_augmentor=True, is_training=True):


        self.use_augmentor = use_augmentor
        self.is_training = is_training

        self.imgs = glob.glob(path + '/**/**/*.png')

        # self.p = Augmentor.Pipeline()

        if use_augmentor:

            self.p = self._augmentor()
            # self.transforms = transforms.Compose([ self.p.torch_transform(), transforms.ToTensor() ])

        
        self.pre_process = transforms.Compose([ transforms.ToTensor(), 
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])


    def __len__(self):


        return len(self.imgs)


    def __getitem__(self, i):
        
        
        img = Image.open(self.imgs[i]).convert('RGB')

        if self.is_training:
            
            # im = self.transforms(img)
            
            # im = self.p._execute_with_array( np.array(img) )

            img = self._transforms([img])
            
            
        ims = [self.pre_process(x) for x in img]

        return np.array(ims[0])



    def _augmentor(self):

        p = Augmentor.Pipeline()

        p.rotate(probability=0.5, max_left_rotation=10, max_right_rotation=10)

        return p


    def _transforms(self, images=[]):
        
        if not isinstance(images, list):
            raise RuntimeError

        params = transforms.RandomCrop.get_params(images[0], output_size=[224, 224]) ## get parameters
        # image = transformsF.crop(image, *params)
        # mask = transformsF.crop(mask, *params)
        # return image, mask

        # images = list( map(transformsF.crop, params) )


        images = [ transformsF.crop(im, *params) for im in images ]

        
        return images


if __name__ == '__main__':


    dataset = DatasetX()

    
    dataset_loader = data.DataLoader(dataset, batch_size=3, num_workers=1) 

    for d in dataset_loader:

        print(d.size())


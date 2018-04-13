from torchvision import transforms
import torch.utils.data as data 
import Augmentor
import glob 
from PIL import Image
import numpy as np 


class DatasetX(data.Dataset):


    def __init__(self, path='/home/wenyu/Desktop/test', use_augmentor=True, is_training=True):


        self.use_augmentor = use_augmentor
        self.is_training = is_training

        self.imgs = glob.glob(path + '/**/**/*.png')


        if use_augmentor:

            self.p = self._augmentor()
            self.transforms = transforms.Compose([ self.p.torch_transform(), transforms.ToTensor() ])


    def __len__(self):


        return len(self.imgs)


    def __getitem__(self, i):
        
        
        img = Image.open(self.imgs[i]).convert('RGB')


        if self.is_training:
            
            # im = self.transforms(img)
            im = self.p._execute_with_array( np.array(img) )
            
                        
        else:

            pass


        return im



    def _augmentor(self):

        p = Augmentor.Pipeline()

        p.rotate(probability=0.5, max_left_rotation=10, max_right_rotation=10)

        return p




if __name__ == '__main__':


    dataset = DatasetX()

    
    dataset_loader = data.DataLoader(dataset, batch_size=3, num_workers=1) 

    for d in dataset_loader:

        print(d.size())


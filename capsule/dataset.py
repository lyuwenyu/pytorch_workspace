import torch
from torchvision.datasets.mnist import MNIST
import torch.utils.data as data
import torchvision.transforms as transforms
import random
from Augmentor import Pipeline

class Dataset(data.Dataset):

    def __init__(self, train=True, root='../data'):
        
        if train:
            self.dataset = MNIST(root=root, train=True)
        else:
            self.dataset = MNIST(root=root, train=False)

        self.transform = transforms.Compose( [transforms.ToTensor()] ) 

        self.p = Pipeline()
        self.p.random_distortion(probability=0.5, grid_width=7, grid_height=7, magnitude=1)


    def __len__(self,):
        return len(self.dataset)

    
    def __getitem__(self, i):
        
        img = self.dataset[i][0]
        lab = self.dataset[i][1]

        # img = self._sample([img], self.p)[0]
        img = self.transform(img)

        # img = 2 * (img - 0.5) 
        lab = torch.eye(10)[lab]

        return img, lab



    def _sample(self, image, p):
        for op in p.operations:
            r = round(random.uniform(0, 1), 1)
            if r <= op.probability:
                image = op.perform_operation(image)
        return image


if __name__ == '__main__':
    
    dataset = Dataset(train=False)

    data_loader = data.DataLoader(dataset, batch_size=5)


    for i, (x, y) in enumerate(data_loader):
        print(x.size())


    # dataset = MNIST(root='../data', train=True)
    # print(dataset[30])
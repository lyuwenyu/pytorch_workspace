import torch
from torchvision.datasets.mnist import MNIST
import torch.utils.data as data
import torchvision.transforms as transforms

class Dataset(data.Dataset):

    def __init__(self, train=True):
        
        if train:
            self.dataset = MNIST(root='../data', train=True)
        else:
            self.dataset = MNIST(root='../data', train=False)

        self.transform = transforms.Compose( [transforms.ToTensor()] ) 

    def __len__(self,):
        return len(self.dataset)

    
    def __getitem__(self, i):
        
        img = self.dataset[i][0]
        lab = self.dataset[i][1]

        img = self.transform(img)
        lab = torch.eye(10)[lab]

        return img, lab



if __name__ == '__main__':
    
    dataset = Dataset(train=False)

    data_loader = data.DataLoader(dataset, batch_size=5)


    for i, (x, y) in enumerate(data_loader):
        print(x.size())



    # dataset = MNIST(root='../data', train=True)
    # print(dataset[30])
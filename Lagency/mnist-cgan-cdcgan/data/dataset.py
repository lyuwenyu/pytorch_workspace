import torch
import torch.utils.data as data
from torchvision.datasets.mnist import MNIST as MNISTx
import torchvision.transforms as transforms
# from tensorflow.examples.tutorials.mnist import input_data


class MNIST(data.Dataset):

    def __init__(self, train=True, root='/home/lvwenyu/workspace/pytorch/data'):

        if train:
            self.dataset = MNISTx(root=root, train=True)
        else:
            self.dataset = MNISTx(root=root, train=False)
        self.transform = transforms.Compose( [transforms.ToTensor()] )


    def __len__(self,):
        return len(self.dataset)


    def __getitem__(self, i):

        img = self.dataset[i][0]
        lab = self.dataset[i][1]

        img = self.transform(img)
        img = 2 * (img - 0.5)

        # lab = torch.eye(10)[lab]

        return img, lab


if __name__ == '__main__':

    dataset = MNIST()

    dataloader = data.DataLoader(dataset, batch_size=10, shuffle=True)


    for xx, yy in dataloader:

        print(xx.size(), yy.size())


        print(xx)

        break
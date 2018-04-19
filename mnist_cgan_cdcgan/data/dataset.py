import torch
import torch.utils.data as data

from tensorflow.examples.tutorials.mnist import input_data


class MNIST(data.Dataset):

    def __init__(self,):

        mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

        self.imgs = mnist.train.images
        self.labs = mnist.train.labels

    def __len__(self):

        return self.imgs.shape[0]


    def __getitem__(self, i):

        # (self.imgs[i]-0.5)/0.5
        return (self.imgs[i]-0.5)/0.5, self.labs[i]




if __name__ == '__main__':

    dataset = MNIST()

    dataloader = data.DataLoader(dataset, batch_size=10, shuffle=True)


    for xx, yy in dataloader:

        print(xx.size(), yy.size())
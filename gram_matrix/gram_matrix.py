

import torch 
import torch.nn as nn 
from torch.autograd import Variable

__all__ = [ 'gram_matrix' ]




class gram_matrix(nn.Module):
    
    def __init__(self,):
        super(gram_matrix, self).__init__()
        
        pass


    def forward(self, x):

        n, c, h, w = x.size()

        x = x.view(n, c, -1)
        x_t = x.permute(0, 2, 1)
        
        return x.bmm(x_t) / (c * h * w)





if __name__ == '__main__':

    x = Variable( torch.rand(3, 10, 7, 7) )
    gram = gram_matrix()


    print(gram(x).size())


import torch
import torch.nn as nn
print(torch.__version__)

class Capsule(nn.Module):
    def __init__(self, in_channel=1, ):
        super(Capsule, self).__init__()

        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channel, 256, kernel_size=9, stride=1),
                        nn.ReLU()
                    )

    
    def forward(self, x):
        
        x = self.conv1(x)


        return x



if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(1,1,28,28, device=device)

    capsule = Capsule(1)
    capsule.to(device)

    out = capsule(x)

    print(out.shape)

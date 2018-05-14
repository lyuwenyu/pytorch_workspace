import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import config

print(torch.__version__)


def squash(tensor, dim=-1):
    squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1+squared_norm)
    return scale*tensor / torch.sqrt(squared_norm)


def init_param(m):
    if isinstance(m, nn.Conv2d):
        # init.xavier_normal_(m.weight)
        init.kaiming_normal(m.weight, mode='fan_in')
        print('init conv2d layer')

    elif isinstance(m, CapsuleLayer):
        for p in m.parameters():
            # init.xavier_normal_(p)
            init.kaiming_normal(p)
        print('init capsule layer')

def param_num(m):
    n = 0
    for p in m.parameters():
        n += p.numel()
    return n


class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules, input_channels, output_channels, kernel_size, stride):
        super(PrimaryCaps, self).__init__()

        self.out_channels = output_channels
        self.num_capsules = num_capsules
        # self.capsules = nn.ModuleList([nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in range(num_capsules)])
        self.capsule = nn.Conv2d(input_channels, output_channels*num_capsules, kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        
        # outputs = [cap(x).view(x.size(0), -1, 1) for cap in self.capsules]
        # outputs = torch.cat(outputs, dim=-1)
        outputs = self.capsule(x).view(x.size(0), self.num_capsules, -1).contiguous().permute(0, 2, 1)
        outputs = squash(outputs)

        return outputs


class CapsuleLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_capsules, num_route_nodes):
        super(CapsuleLayer, self).__init__()
        # self.num_iterations = config.route_iterations
        self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, input_dim, output_dim))
        self.capsule_bias = nn.Parameter(torch.zeros(num_capsules, 1, 1, output_dim))

    def forward(self, x):
        
        priors =  x[:, None, :, None, :] @ self.route_weights[None, :, :, :, :]
        B = torch.zeros(*priors.size()[:3], 1, 1, requires_grad=False).to(x.device)
        # B = torch.zeros(*priors.size(), requires_grad=False).to(x.device)

        for _ in range(config.route_iterations):
            
            probs = F.softmax(B, dim=2)
            outputs = squash( (priors*probs).sum(dim=2, keepdim=True) ) + self.capsule_bias
            # delta_b = (priors*outputs).sum(dim=-1, keepdim=True)
            # print(-((priors-outputs)**2).sum(dim=-1, keepdim=True)/config.sigma**2)
            delta_b = torch.exp(-((priors-outputs)**2).sum(dim=-1, keepdim=True)/config.sigma**2)
            B = B + delta_b

        return outputs.squeeze()



class CapsuleLoss(nn.Module):
    def __init__(self, ):
        super(CapsuleLoss, self).__init__()

    def forward(self, logits, labels):

        left = F.relu(0.9-logits) ** 2
        right = F.relu(logits-0.1) ** 2
        margin_loss = labels * left + 0.5 * (1.0 - labels) * right

        return margin_loss.sum() / logits.size()[0]


class CapsuleNet(nn.Module):
    def __init__(self, in_channels=1, ):
        super(CapsuleNet, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=9, stride=1)
        self.primcap = PrimaryCaps(8, 256, 32, 9, 2)
        self.capsule = CapsuleLayer(8, 16, 10, 1152)

        # self.apply(init_param)

    def forward(self, x):

        x = F.relu(self.conv(x), inplace=True)
        x = F.relu(self.primcap(x), inplace=True)
        x = self.capsule(x)

        logits = (x**2).sum(dim=-1)**0.5
        logits = F.softmax(logits, dim=-1)

        return logits

    
        

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = CapsuleNet()
    net = net.to(device)
    x = torch.randn(64, 1, 28, 28)
    x = x.to(device)

    out = net(x)

    # print(out.size())
    # print(out)


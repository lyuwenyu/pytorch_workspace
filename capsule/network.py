import torch
import torch.nn as nn
import torch.nn.functional as F
print(torch.__version__)


def squash(tensor, dim=-1):
    squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1+squared_norm)
    return scale*tensor / torch.sqrt(squared_norm)


class PrimaryCaps(nn.Module):
    def __init__(self, capsules, input_channels, output_channels, kernel_size, stride):
        super(PrimaryCaps, self).__init__()

        self.capsules = nn.ModuleList([ nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in range(capsules)])
        
    def forward(self, x):
        
        outputs = [cap(x).view(x.size(0), -1, 1) for cap in self.capsules]
        outputs = torch.cat(outputs, dim=-1)
        outputs = squash(outputs)

        return outputs

    def param_num(self,):
        n = 0
        for p in self.parameters():
            n += p.numel()
        return n


class CapsuleLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_capsules, num_route_nodes):
        super(CapsuleLayer, self).__init__()
        
        self.num_iterations = 3
        self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, input_dim, output_dim)) 

    def forward(self, x):
        
        priors =  x[:, None, :, None, :] @ self.route_weights[None, :, :, :, :]
        B = torch.zeros(*priors.size()[:3], 1, 1, requires_grad=False)

        for _ in range(self.num_iterations):
            
            probs = F.softmax(B, dim=2)
            outputs = squash( (priors*probs).sum(dim=2, keepdim=True) )
            
            delta_c = (priors*outputs).sum(dim=-1, keepdim=True)  #
            # t = priors - outputs
            # k = torch.exp(torch.dot(t, t)) / (sigma**2)
            B = B + delta_c

        return outputs.squeeze()



class CapsuleLoss(nn.Module):
    def __init__(self, ):
        super(CapsuleLoss, self).__init__()

    def forward(self, logits, labels):

        left = F.relu(0.9-logits) ** 2
        right = F.relu(logits-0.1) ** 2
        margin_loss = labels * left + 0.5 * (1.0 - labels) * right

        return margin_loss.mean()



class CapsuleNet(nn.Module):
    def __init__(self, ):
        super(CapsuleNet, self).__init__()

        self.conv = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1)
        self.primcap = PrimaryCaps(8, 256, 32, 9, 2)
        self.capsule = CapsuleLayer(8, 16, 10, 1152)    

    def forward(self, x):

        x = self.conv(x)
        x = self.primcap(x)
        x = self.capsule(x)

        logits = (x**2).sum(dim=-1)**0.5
        logits = F.softmax(logits, dim=-1)

        return logits


if __name__ == '__main__':

    # device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

    net = CapsuleNet()
    x = torch.randn(64, 1, 28, 28)

    out = net(x)

    print(out.size())
    print(out)


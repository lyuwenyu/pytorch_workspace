import torch
import torch.nn as nn
import torch.nn.functional as F


class M(nn.Module): # (torch.jit.ScriptModule):
    def __init__(self,):
        super(M, self).__init__()

        self.mm = nn.Conv2d(1, 3, 3, 2, 0)

    # @torch.jit.script_method
    def forward(self, x):
        '''
        '''
        out = self.mm(x)

        return out

m = M()
data = torch.rand(1, 1, 10, 10)
out = m(data)
print(out.shape, out.sum().item())

# 
traced_script_module = torch.jit.trace(m, data)
traced_script_module.save('test.pt')



# --------------
print('-------')
# --------------

class MM(torch.jit.ScriptModule):
    def __init__(self,):
        super(MM, self).__init__()

        self.mm = torch.jit.trace(nn.Conv2d(1, 3, 3, 2, 0), torch.rand(1, 1, 10, 10))

    @torch.jit.script_method
    def forward(self, x):
        '''
        '''
        out = F.relu(self.mm(x))

        return out

m = MM()
out = m(data)
print(out.shape, out.sum().item())

# 
traced_script_module = torch.jit.trace(m, data)
traced_script_module.save('test.pt')
import torch
import torch.nn as nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# torch.autograd.Variable legacy

# new tensor style
a = torch.randn([1,2,3], device=device, requires_grad=True, dtype=torch.float32)
a = a.to(torch.int32)

print(a)
print(a.shape)
print(a.requires_grad)
print(a.device)
print(a.dtype)

# scale
b = torch.tensor(1)

print(b)
print(b.item())
print(b.dtype)

# context
with torch.no_grad():
    # inference
    pass

# 
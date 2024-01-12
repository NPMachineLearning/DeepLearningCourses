import torch

a = torch.tensor([
    [1,2],
    [3,4],
    [5,6]
])
b = torch.tensor([
    [1,2,3],
    [4,5,6]
])

c = torch.mm(a, b)
print(c, c.shape)

c = a.mm(b)
print(c, c.shape)
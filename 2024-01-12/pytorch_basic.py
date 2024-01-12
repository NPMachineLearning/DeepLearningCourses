import torch

a = torch.tensor(10)
print(a, a.dtype)
b = torch.tensor(10, dtype=torch.int32)
print(b)
c = torch.tensor(10.)
print(c, c.dtype)
d = torch.tensor(10., dtype=torch.float64)
print(d)

e = torch.tensor([1,2,3])
print(e)
f = torch.tensor([[1,2,3], [4,5,6]])
print(f)
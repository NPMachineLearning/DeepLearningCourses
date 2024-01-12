import torch

a = torch.DoubleTensor([1,2,3])
print(a, a.dtype)

b = torch.DoubleTensor(10)
print(b)

c = torch.zeros(10)
print(c, c.dtype)

c = torch.zeros(3, 10)
print(c)
c = torch.zeros([3, 10])
print(c)
c = torch.zeros([10], dtype=torch.int32)
print(c)

d = torch.ones([10])
print(c)

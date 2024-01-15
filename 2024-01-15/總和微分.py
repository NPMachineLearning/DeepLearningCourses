import torch

a = torch.tensor([0.1, 0.2, 0.3, 0.4])
b = torch.tensor([1., 2., 3., 4.])

x = torch.tensor([2.], requires_grad=True)
s = torch.sum(a*x**2+b)
s.backward()
print(x.grad)
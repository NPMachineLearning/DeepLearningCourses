import torch

x = torch.tensor(2., requires_grad=True)
y = 0.6 * torch.square(x) + 5
y.backward()
print(x.grad)
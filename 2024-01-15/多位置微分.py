import torch

t = torch.tensor([1.,2.,3.,4.,5.], requires_grad=True)
g = 9.8
s = 0.5*g*t**2
s.backward(gradient=torch.ones(t.shape[0]))
print(t.grad)
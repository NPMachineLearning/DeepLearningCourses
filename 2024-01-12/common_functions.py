import torch
import numpy as np

a = torch.linspace(0, 10.5, 10)
print(a)

b = np.linspace(0, 10.5, 10)
print(b)

c = list(range(1, 10, 1))
print(c)

f = list(torch.range(1, 10, 1.5))
print(f)

g = torch.arange(0, 10, 1.5)
print(g)
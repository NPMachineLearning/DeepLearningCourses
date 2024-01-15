import torch
import numpy as np
import pylab as plt

t = np.linspace(0, 10, 11)
g = 9.8
s = 0.5*g*t**2
plt.xlim(0, 20)
plt.ylim(0, 500)
plt.plot(t, s)
plt.show()

t = torch.tensor(2.5, requires_grad=True)
s = 0.5*g*t**2
s.backward()
print(t.grad)

t = torch.tensor(5., requires_grad=True)
s = 0.5*g*t**2
s.backward()
print(t.grad)
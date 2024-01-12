import torch
import matplotlib.pyplot as plt

a = torch.randint(0, 5, [2, 10])
print(a)
b= torch.rand([2, 1000])
print(b)

plt.scatter(b[0], b[1], s=1)
plt.show()

c= torch.randn([2, 1000])
print(b)

plt.scatter(c[0], c[1], s=1)
plt.show()

d = torch.normal(50, 100, [2, 1000])

plt.scatter(d[0], d[1], s=1)
plt.show()
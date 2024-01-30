import numpy as np
from matplotlib import pyplot as plt

a = range(20)
b = range(20)

x, y = np.meshgrid(a, b)
# x = x.reshape(-1)
# y = y.reshape(-1)
print(x)
print(y)
plt.scatter(x, y, s=0.5, c="b")
plt.show()
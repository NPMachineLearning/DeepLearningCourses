import numpy as np
from matplotlib import pyplot as plt

x = np.linspace(0, 100, 1000)
y=0.001*(x**6)-0.1*(x**5)-0.68*(x**4)+10000*(x**2)+2
plt.plot(x,y)
plt.show()
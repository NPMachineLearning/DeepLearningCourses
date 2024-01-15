import numpy as np
import pylab as plt

n = 30
x = np.linspace(20, 40, n)
noise = (np.random.random(n)*2-1)*40
y = 500-x*10+noise
plt.scatter(x, y, c="blue")

# args = np.polyfit(x, y, 1)
#deg 1: ax+b
#deg 2: ax^2+bx+c
#deg 3: ax^3+bx^2+cx+d

# y = args[0]*x+args[1]
# y = args[0]*x**2+args[1]*x+args[2]
plt.plot(x, y)

f = np.poly1d(np.polyfit(x,y,100))
plt.plot(x, f(x))

plt.show()
import pylab as plt
import numpy as np

fig = plt.figure(figsize=(9, 6))
ax = plt.axes()
np.random.seed(32)
n = 20
x = np.linspace(-10, 10, n)
y = 0.5*x+3+np.random.randint(-5, 5, n)
a = 0
b = 0
lr = 2.5e-3
r = a*x+b
diff = 1000000
pre_loss = 100000
i = 0
while diff > 0.001:
    i+=1
    ax.clear()
    ax.set_xlim(-15, 15)
    ax.set_ylim(-8, 15)
    ax.plot([-15, 15], [0,0], c="k", linewidth=0.5)
    ax.plot([0,0], [-8, 15], c="k", linewidth=0.5)
    ax.scatter(x,y,c="g")
    f = np.poly1d(np.polyfit(x,y,1))
    ax.plot(x, f(x), c="b", linewidth=3)
    da = np.sum((r-y)*x)
    db = np.sum(r-y)
    a = a-da*lr
    b = b-db*lr
    r = a*x+b
    loss = np.sum(np.square(y-r))
    diff = pre_loss - loss
    pre_loss = loss
    ax.plot(x, r, c="r", linewidth=1)
    plt.pause(0.1)
plt.show()


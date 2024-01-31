from matplotlib import pyplot as plt

from Regression import *
from SGD import SGD

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
epochs = 150
x, y = get_data(100)
mesh, contour = getContour(x, y)
ax2 = ax[1].contourf(mesh[0], mesh[1], contour, 15, cmap=plt.cm.Purples)
ax[1].set_xlabel("a")
ax[1].set_ylabel("b")
plt.colorbar(ax2, ax=ax[1])
ax[1].set_xlim(-10, 15)
ax[1].set_ylim(-10, 15)
init_a = -9
init_b = -9
ax[1].scatter(init_a, init_b, c="g")
lr = 0.01
gd = SGD(init_a, init_b, x, y, lr)
for epoch in range(epochs):
    gd.update()
    ax[0].clear()
    ax[0].set_xlim(-5, 5)
    ax[0].set_ylim(-30, 30)
    ax[0].scatter(x, y)
    ax[0].plot([x[0], x[-1]], [gd.a*x[0]+gd.b, gd.a*x[-1]+gd.b], c="orange")
    f = np.poly1d(np.polyfit(x,y,1))
    ax[0].plot(x, f(x), c="green")
    ax[1].plot([gd.old_a, gd.a], [gd.old_b, gd.b], c="r")
    ax[1].scatter(gd.a, gd.b, c="green")
    plt.pause(0.1)
plt.show()
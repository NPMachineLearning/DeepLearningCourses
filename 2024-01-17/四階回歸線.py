import math

import pylab as plt
import numpy as np
import torch

def f_loss(a,b,c,d,e):
    return torch.sum(
        torch.square(
            ty - (
                a * tx.pow(4) +
                b * tx.pow(3) +
                c * tx.pow(2) +
                d * tx.pow(1) +
                e * tx.pow(0)
            )
        )
    )

fig = plt.figure(figsize=(9, 6))
ax = plt.axes()
np.random.seed(32)
n = 20
x = np.linspace(-10, 10, n)
y = 0.5*x+3+np.random.randint(-5, 5, n)
tx = torch.tensor(x)
ty = torch.tensor(y)
lr = 2.5e-3
diff = 1000000
pre_loss = 100000
a = torch.tensor([0.], requires_grad=True)
b = torch.tensor([0.], requires_grad=True)
c = torch.tensor([0.], requires_grad=True)
d = torch.tensor([0.], requires_grad=True)
e = torch.tensor([0.], requires_grad=True)
i = 0
while diff > 0.001:
    i += 1
    ax.clear()
    ax.set_xlim(-15, 15)
    ax.set_ylim(-8, 15)
    ax.plot([-15, 15], [0, 0], c="k", linewidth=0.5)
    ax.plot([0, 0], [-8, 15], c="k", linewidth=0.5)
    ax.scatter(x, y, c="g")
    f = np.poly1d(np.polyfit(x, y, 4))
    ax.plot(x, f(x), c="b", linewidth=3)
    f_loss(a,b,c,d,e).backward()
    with torch.no_grad():
        a -= a.grad * lr * 1e-6
        b -= b.grad * lr * 1e-5
        c -= c.grad * lr * 1e-3
        d -= d.grad * lr * 1e-1
        e -= e.grad * lr
        ax.plot(x,
                a.numpy()[0] * np.power(x, 4) +
                b.numpy()[0] * np.power(x, 3) +
                c.numpy()[0] * np.power(x, 2) +
                d.numpy()[0] * np.power(x, 1) +
                e.numpy()[0] * np.power(x, 0),
                c="r",
                linewidth=1
                )
    l = torch.sum(torch.square(ty - torch.pow(a, 4) + torch.pow(b, 3) + torch.pow(c, 2) + torch.pow(d, 1) + e))
    diff = math.fabs(pre_loss-l)
    pre_loss = l
    a.grad.zero_()
    b.grad.zero_()
    c.grad.zero_()
    d.grad.zero_()
    e.grad.zero_()
    plt.pause(0.1)
plt.show()
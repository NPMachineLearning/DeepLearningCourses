import math

import pylab as plt
import numpy as np
import torch
def f_loss(params):
    str_f = 'torch.mean(torch.square(ty - ('
    for i in range(order+1):
        str_f += f"params[{i}]* tx.pow({order-i}) +"
    str_f = str_f[:-1] + ")))"
    return eval(str_f)

fig = plt.figure(figsize=(9, 6))
ax = plt.axes()
np.random.seed(32)
n = 20
x = np.linspace(-10, 10, n)
y = 0.5*x+3+np.random.randint(-5, 5, n)
tx = torch.tensor(x)
ty = torch.tensor(y)
order = 5
scale = [1e-7, 1.8e-5, 1.2e-3, 5.0e-4, 20.e-3, 1.2e2]
lr = 2.6e-3
diff = 1000000
pre_loss = 100000
params = [torch.tensor([0.], requires_grad=True) for i in range(order+1)]
epochs = 1500
for epoch in range(epochs):
    ax.clear()
    ax.set_xlim(-15, 15)
    ax.set_ylim(-8, 15)
    ax.plot([-15, 15], [0, 0], c="k", linewidth=0.5)
    ax.plot([0, 0], [-8, 15], c="k", linewidth=0.5)
    ax.scatter(x, y, c="g")
    f = np.poly1d(np.polyfit(x, y, 5))
    ax.plot(x, f(x), c="b", linewidth=3)

    f_loss(params).backward()
    with torch.no_grad():
        for i in range(order+1):
            params[i] -= params[i].grad * lr * scale[i]
        str_f = ""
        for i in range(order+1):
            str_f += f"params[{i}].numpy()[0] * np.power(x, {order-i}) +"
        str_f = str_f[:-1]
        ax.plot(x, eval(str_f), c="r", linewidth=1)
    for i in range(order+1):
        params[i].grad.zero_()
    plt.pause(0.01)
plt.show()


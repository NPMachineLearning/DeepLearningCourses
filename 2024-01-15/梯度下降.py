import numpy as np
import torch
import pylab as plt

def f(x):
    if torch.is_tensor(x):
        return x.square()
    else:
        return np.square(x)

x = np.linspace(-5, 5, 100)
y = f(x)

fig = plt.figure(figsize=(9,7))
ax = fig.subplots()
epochs = 100
lr = 0.1
tx = torch.tensor([-5.], requires_grad=True)
history = []
# x_{t+1} = x_{t}-f'(x_t) * lr
for i in range(epochs):
    f(tx).backward()
    with torch.no_grad():
        now_x = tx.numpy()[0]
        history.append(now_x)
        # get gradient
        a = tx.grad.numpy()[0]
        # y = ax + b
        # b = y - ax
        b = f(now_x)-a * now_x
        ax.clear()
        plt.plot(x, y, c="b")
        plt.scatter(history, f(history), c="r")
        ax.set_xlim(-10, 10)
        ax.set_ylim(-2, 35)
        xl = now_x - 2
        xr = now_x + 2
        # slop line
        plt.plot([xl, xr], [a*xl+b, a*xr+b], c="orange")
        tx -= tx.grad * lr
        tx.grad.zero_()
        plt.pause(0.01)
plt.show()

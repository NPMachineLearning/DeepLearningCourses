# y = x^4 - 60x^3 - x + 1
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return (np.power(x, 4) - 60*np.power(x, 3) - x + 1) / shrink
def df(x):
    return (4*np.power(x,3) - 180*np.power(x, 2) - 1) / shrink
def bias(a, x):
    return f(x) - a * x

shrink = 1e6

x = np.linspace(-30, 60, 100)
y = f(x)
current_x = x[0]
history = [current_x]
epochs = 500
lr = 35
fig = plt.figure(figsize=(10, 6))
ax = plt.axes()
for epoch in range(epochs):
    ax.clear()
    ax.set_xlim(-45, 70)
    ax.set_ylim(-2, 3)
    ax.plot(x, y)
    ax.scatter(history, f(history), c="red")
    a = df(current_x)
    b = bias(a, current_x)
    xl = current_x - 3
    xr = current_x + 3
    line_x = [xl, xr]
    line_y = [a * xl + b, a * xr + b]
    ax.plot(line_x, line_y, c="orange")
    current_x -= a * lr
    history.append(current_x)
    plt.pause(0.01)
plt.show()
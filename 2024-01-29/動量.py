import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return np.square(x)
def df(x):
    return 2*x
def bias(a, x):
    return f(x) -a * x

x = np.linspace(-5 , 5, 100)
y = f(x)

epochs = 500
fig = plt.figure(figsize=(10, 7))
ax = plt.axes()
current_x = -5
history = [current_x]
lr = 0.2
decay = 0.01
v = 0
mu = 0.9
for epoch in range(epochs):
    ax.clear()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-5, 30)
    ax.plot(x, y)
    ax.scatter(history, f(history), c="red")
    a = df(current_x)
    b = bias(a, current_x)
    xl = current_x - 3
    xr = current_x + 3
    line_x = [xl, xr]
    line_y = [a * xl + b, a * xr + b]
    ax.plot(line_x, line_y, c="orange")
    v = a * lr + mu * v
    current_x -= v
    history.append(current_x)
    plt.pause(0.01)
plt.show()
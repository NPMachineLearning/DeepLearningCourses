import numpy as np

np.random.seed(21)

def get_data(n):
    x = np.arange(-5, 5.1, 10/n)
    y = 3*x + 2 + (np.random.rand(len(x))-0.5)*20
    return x, y

def getContour(x, y):
    a = np.arange(-10, 16, 1)
    b = np.arange(-10, 16, 1)
    mesh = np.meshgrid(a, b)
    loss = np.zeros([len(a), len(b)])
    for px, py in zip(x, y):
        loss += ((mesh[0]*px + mesh[1])-py)**2
    return mesh, loss
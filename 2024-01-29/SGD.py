import numpy as np

from BGD import BGD


class SGD(BGD):
    def __init__(self, a, b, x, y, lr):
        super().__init__(a,b,x,y,lr)
    def gradient(self):
        idx = np.random.randint(len(self.x))
        grad_a = 2 * (self.a * self.x[idx] + self.b - self.y[idx]) * self.x[idx]
        grad_b = 2 * (self.a * self.x[idx] + self.b - self.y[idx])
        return grad_a, grad_b
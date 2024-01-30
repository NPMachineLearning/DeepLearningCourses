import numpy as np


class BGD():
    def __init__(self, a, b, x, y, lr):
        super().__init__()
        self.a = a
        self.b = b
        self.x = x
        self.y = y
        self.lr = lr
        self.old_a = a
        self.old_b = b
        self.loss = None
    def gradient(self):
        grad_a = 2*np.mean((self.a * self.x + self.b - self.y)*self.x)
        grad_b = 2*np.mean((self.a * self.x + self.b - self.y))
        return grad_a, grad_b
    def update(self):
        grad_a, grad_b = self.gradient()
        self.old_a = self.a
        self.old_b = self.b
        self.a = self.a - self.lr * grad_a
        self.b = self.b - self.lr * grad_b
        self.loss = ((self.a * self.x + self.b) - self.y)**2
        self.loss = np.mean(self.loss)
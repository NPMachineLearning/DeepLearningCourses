import numpy as np

from MBGD import MBGD


class SGDM(MBGD):
    def __init__(self, a, b, x, y, lr, batch_size, gamma):
        super().__init__(a, b, x, y, lr, batch_size)
        self.gamma = gamma
        self.ma = 0
        self.mb = 0
    def update(self):
        grad_a, grad_b = self.gradient()
        self.old_a = self.a
        self.old_b = self.b
        self.ma = self.gamma * self.ma + self.lr * grad_a
        self.mb = self.gamma * self.mb + self.lr * grad_b
        self.a = self.a - self.ma
        self.b = self.b - self.mb
        self.loss = ((self.a * self.x + self.b) - self.y)**2
        self.loss = np.mean(self.loss)
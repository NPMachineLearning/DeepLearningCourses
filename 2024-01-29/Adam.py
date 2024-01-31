import numpy as np

from MBGD import MBGD


class Adam(MBGD):
    def __init__(self, a, b, x, y, lr, batch_size, beta1, beta2):
        super().__init__(a, b, x, y, lr, batch_size)
        self.beta1 = beta1
        self.beta2 = beta2
        self.e = 1e-6
        self.sum_ma = 0
        self.sum_mb = 0
        self.sum_grad_a = 0
        self.sum_grad_b = 0
    def update(self):
        grad_a, grad_b = self.gradient()
        self.old_a = self.a
        self.old_b = self.b
        self.sum_grad_a = self.beta1 * self.sum_grad_a + (1 - self.beta1) * grad_a ** 2
        self.sum_grad_b = self.beta2 * self.sum_grad_b + (1 - self.beta2) * grad_b ** 2
        self.a = self.a - (self.lr / (np.sqrt(self.sum_grad_a) + self.e)) * grad_a
        self.b = self.b - (self.lr / (np.sqrt(self.sum_grad_b) + self.e)) * grad_b
        loss = ((self.a * self.x + self.b) - self.y) ** 2
        self.loss = np.mean(loss)
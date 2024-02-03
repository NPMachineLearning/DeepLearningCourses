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
    def update(self, t):
        grad_a, grad_b = self.gradient()
        self.old_a = self.a
        self.old_b = self.b

        self.sum_ma = self.beta1 * self.sum_ma + (1 - self.beta1) * grad_a
        self.sum_mb = self.beta1 * self.sum_mb + (1 - self.beta1) * grad_b

        self.sum_grad_a = self.beta2 * self.sum_grad_a + (1 - self.beta2) * grad_a ** 2
        self.sum_grad_b = self.beta2 * self.sum_grad_b + (1 - self.beta2) * grad_b ** 2

        ma = self.sum_ma / (1 - np.power(self.beta1, t))
        mb = self.sum_mb / (1 - np.power(self.beta1, t))

        va = self.sum_grad_a / (1 - np.power(self.beta2, t))
        vb = self.sum_grad_b / (1 - np.power(self.beta2, t))

        self.a -= (self.lr * ma) / (np.sqrt(va) + self.e)
        self.b -= (self.lr * mb) / (np.sqrt(vb) + self.e)

        loss = ((self.a * self.x + self.b) - self.y)**2
        self.loss = np.mean(loss)
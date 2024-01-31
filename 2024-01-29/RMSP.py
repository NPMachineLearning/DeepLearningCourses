import numpy as np

from Adagrad import Adagrad


class RMSP(Adagrad):
    def __init__(self, a, b, x, y, lr, batch_size, rho):
        super().__init__(a, b, x, y, lr, batch_size)
        self.rho = rho
    def update(self):
        grad_a, grad_b = self.gradient()
        self.old_a = self.a
        self.old_b = self.b
        self.sum_grad_a = self.rho * self.sum_grad_a + (1 - self.rho) * grad_a**2
        self.sum_grad_b = self.rho * self.sum_grad_b + (1 - self.rho) * grad_b**2
        self.a = self.a - (self.lr / (np.sqrt(self.sum_grad_a) + self.e)) * grad_a
        self.b = self.b - (self.lr / (np.sqrt(self.sum_grad_b) + self.e)) * grad_b
        loss = ((self.a * self.x + self.b) - self.y) ** 2
        self.loss = np.mean(loss)
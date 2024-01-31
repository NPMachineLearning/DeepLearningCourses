import numpy as np

from BGD import BGD


class MBGD(BGD):
    def __init__(self, a, b, x, y, lr, batch_size):
        super().__init__(a, b, x, y, lr)
        self.batch_size = batch_size
        self.shuffle = np.random.permutation(len(self.x))
        self.refresh()
        self.update_batch()
    def refresh(self):
        self.shuffle = np.random.permutation(len(self.x))
        self.start = 0
    def update_batch(self):
        idx = self.shuffle[self.start:self.start+self.batch_size]
        self.start += self.batch_size
        self.x_batch = self.x[idx]
        self.y_batch = self.y[idx]
    def gradient(self):
        grad_a = 2 * (np.mean((self.a * self.x_batch + self.b - self.y_batch) * self.x_batch))
        grad_b = 2 * np.mean((self.a * self.x_batch + self.b - self.y_batch))
        if self.start > len(self.x):
            self.refresh()
        self.update_batch()
        return grad_a, grad_b
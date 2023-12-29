from time import time

import numpy as np

a = []
t1 = time()
for i in range(1000):
    a += ["hello"] * 1000
t2 = time()
print(t2 - t1)

a = []
t1 = time()
for i in range(1000):
    a = np.r_[a, ["hello"]*1000]
t2 = time()
print(t2 - t1)
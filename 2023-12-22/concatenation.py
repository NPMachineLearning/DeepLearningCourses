import numpy as np

a = [1,2,3,4,5]
b = [6,7,8,9,10]
c = np.c_[a, b]
c_r = np.r_[a, b]
print(c)
print(c_r)
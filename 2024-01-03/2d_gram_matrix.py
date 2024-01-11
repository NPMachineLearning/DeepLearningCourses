import numpy as np

a = np.array([[1,2,3,4,5], [6,7,8,9,10], [11,12,13,14,15]])
b = a.T
print(a)
print(b)
b = b.ravel()
print(b)
print(np.outer(a, b))
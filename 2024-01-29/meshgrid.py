import numpy as np

a = range(5)
b = range(5)

x, y = np.meshgrid(a, b)
print(x, y)

for row in zip(x, y):
    for col in zip(row[0], row[1]):
        print(col)
import time

import pylab as plt
import numpy as np

# Monte-Carlo method: Calculating for PI where PI is area of circle
# thus quart of circle is PI/4, formula: area = PI/4 then PI = 4 * PI

# batch = 10000
#
# X = np.random.uniform(0,1, batch)
# y = np.random.uniform(0,1, batch)
#
# plt.scatter(X, y, s=1)
# plt.show()

batch = 100000000
epochs = 200

for epoch in range(epochs):
    t1 = time.time()
    points = [
        np.random.uniform(0, 1, batch),
        np.random.uniform(0, 1, batch)
    ]
    dist = np.sqrt(np.square(points[0]) + np.square(points[1]))
    count = np.where(dist < 1.0)[0].shape
    print(count)

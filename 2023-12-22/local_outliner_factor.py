import numpy as np
import pylab as plt
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor as LOF

np.random.seed(32)
inliners = np.random.randn(100, 2)
inliners = np.r_[inliners+3, inliners-3]

outliners = np.random.uniform(-6, 6, (20, 2))
data = np.r_[inliners, outliners]

lof = LOF(n_neighbors=20, contamination="auto")
y_pred = lof.fit_predict(data)
scores = lof.negative_outlier_factor_
print(y_pred)
print(scores)
radius = np.ones([len(y_pred)])
radius -= y_pred
print(radius)

X = data[:, 0]
y = data[:, 1]
plt.scatter(X, y, s=5)
plt.scatter(X, y, s=radius*100, edgecolors="r", facecolor="none")
plt.xlim(-7, 7)
plt.ylim(-7, 7)
plt.show()
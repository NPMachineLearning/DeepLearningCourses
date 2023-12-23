import numpy as np
import pandas as pd

X = np.random.randint(1 ,100, 10)
y = np.random.randint(1 ,100, 10)

print(X)
print(y)

X_mean = X.mean()
y_mean = y.mean()

covariance = np.sum((X - X_mean) * (y - y_mean))
print(covariance)
X_d = np.sqrt(np.square(X - X_mean).sum())
y_d = np.sqrt(np.square(y - y_mean).sum())
p = covariance / (X_d * y_d)
print(p)

df = pd.DataFrame({"x":X, "y":y})
print(f"Pearson: {df.corr()}")
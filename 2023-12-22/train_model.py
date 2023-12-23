import pickle

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from get_data import get_data
import numpy as np
import pandas as pd

df = get_data()

data = np.c_[df["LSTAT"], df["RM"]]
X = pd.DataFrame(data, columns=["LSTAT", "RM"])
y = df["PRICE"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# LinearRegression model
lr_model = LinearRegression()

# train model
lr_model.fit(X_train, y_train)

# save model
pickle.dump(lr_model, open("housing_model.pkl", "wb"))

# load model
loaded_model = pickle.load(open("housing_model.pkl", "rb"))

# test model score
print(loaded_model.score(X_test, y_test))

# predict
y_pred = loaded_model.predict(X_test)
for i in zip(y_pred, y_test):
    print(i)

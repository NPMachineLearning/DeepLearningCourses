from get_data import get_data
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

df = get_data()
df = df[["PRICE", "LSTAT", "RM"]]
X = df[["LSTAT", "RM"]]
y = df["PRICE"]
lof = LocalOutlierFactor(20, contamination="auto")
y_pred = lof.fit_predict(X, y)
df = pd.DataFrame(data=df.loc[np.where(y_pred==1)], columns=["PRICE", "LSTAT", "RM"])
print(df)


# improving model score
# adjust skewness close to 0 for LSTAT (weak relationship)
# adjust skewness to much bigger RM (strong relationship)
data = np.c_[df["LSTAT"]**(1/3), df["RM"]**5]
X = pd.DataFrame(data, columns=["LSTAT", "RM"])
y = df["PRICE"]

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# LinearRegression model
lr_model = LinearRegression()

# train model
lr_model.fit(X_train, y_train)

# save model
pickle.dump(lr_model, open("housing_model_lof.pkl", "wb"))

# load model
loaded_model = pickle.load(open("housing_model_lof.pkl", "rb"))

# test model score
print(loaded_model.score(X_test, y_test))
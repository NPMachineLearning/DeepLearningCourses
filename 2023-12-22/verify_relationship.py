from get_data import get_data
import pylab as plt
import seaborn as sns

df = get_data()
features = ["LSTAT", "RM"]

plt.figure(figsize=(15, 7))
for i, col in enumerate(features):
    plt.subplot(1, len(features), i+1)
    plt.scatter(df[col], df["PRICE"])
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel("PRICE")
plt.show()


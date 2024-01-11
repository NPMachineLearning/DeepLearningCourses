from get_data import get_data
import seaborn as sns
import pylab as plt

df = get_data()
print(df)

# visualize price
sns.histplot(df["PRICE"])
plt.show()
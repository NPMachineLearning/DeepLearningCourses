from get_data import get_data
import pylab as plt
import seaborn as sns

df = get_data()
lstat = df["LSTAT"]**(1/3)
sns.histplot(lstat, kde=True)
skewness = round(lstat.skew(), 2)
plt.title(skewness)
plt.show()
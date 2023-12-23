from get_data import get_data
import pylab as plt
import seaborn as sns

df = get_data()
corrmat = df.corr()
print(corrmat)
corrmat = corrmat.nlargest(len(corrmat), columns="PRICE")
sns.set(font_scale=0.7)
sns.heatmap(corrmat, annot=True, annot_kws={"size":8}, fmt=".2f")
plt.show()
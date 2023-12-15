#pip install matplotlib
import time
import numpy as np
import pylab as plt
batch=10000
"""
#uniform : 平均分佈, 在 0, 1之間的機率是一樣的
#normal : 常態分佈, 愈接近0的機會愈大, 愈遠機會愈小
#蒙地卡羅求 pi : 已知1/4圓的面積(area) , pi=4*area
xs=np.random.uniform(0,1,batch)
ys=np.random.uniform(0,1,batch)
plt.scatter(xs, ys, s=1)
plt.show()
"""
batch=100_000_000
incircle=0
epochs=200#世代
for e in range(epochs):
    t1=time.time()
    points=[np.random.uniform(0,1,batch), np.random.uniform(0,1, batch)]
    dist=np.sqrt(np.square(points[0])+np.square(points[1]))#總共算了 1 億次
    count=np.where(dist<=1)[0].shape[0]
    print(count)
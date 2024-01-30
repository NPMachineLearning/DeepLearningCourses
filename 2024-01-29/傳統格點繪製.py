import matplotlib.pyplot as plt

for x in range(20):
    for y in range(20):
        plt.scatter(x, y, s=0.5, c="b")
plt.show()
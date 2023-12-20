import numpy as np

np.random.seed(32)
data = np.random.randint(1, 100, 4)
print(data)
data = np.sort(data)
print(data)

# Mean
print("Mean:", np.mean(data))

# Median if even then add middle-left and middle-right
# divide by 2
# e.g [1,2,3,4] = (2+3)/2
# e.g [1,2,3,4,5] = 3
print("Median:", np.median(data))

# Standard deviation STD
data = np.random.randint(1, 100, 4)
print(data)
sum = 0
for d in data:
    sum += (d-np.mean(data))**2
std = (sum/len(data))**0.5
print(std)
print(np.std(data))

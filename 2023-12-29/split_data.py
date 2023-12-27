# dataset https://mahaljsp.ddns.net/files/vgg19/flowers_17.zip
import os.path
import shutil

import numpy as np

flowers = []
flower_array = []

with open("label.txt") as file:
    for line in file:
        line = line.strip()
        cols = line.split()
        start = int(cols[0])
        end = int(cols[1])
        flower = cols[2]
        flowers.append(flower)
        flower_array += [flower] * (end-start+1)

print(flowers)
print(flower_array)

source = "./flowers_17"
train = "./train_images"
test = "./test_images"
if os.path.exists(train):
    shutil.rmtree(train)
if os.path.exists(test):
    shutil.rmtree(test)
os.mkdir(train)
os.mkdir(test)
for flower in flowers:
    os.mkdir(os.path.join(train, flower))
    os.mkdir(os.path.join(test, flower))

data = list(zip(os.listdir(source), flower_array))
np.random.seed(21)
np.random.shuffle(data)
images, labels = zip(*data)





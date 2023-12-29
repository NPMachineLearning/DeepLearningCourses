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

source_dir = "./flowers_17"
train_dir = "./train_images"
test_dir = "./test_images"
if os.path.exists(train_dir):
    shutil.rmtree(train_dir)
if os.path.exists(test_dir):
    shutil.rmtree(test_dir)
os.mkdir(train_dir)
os.mkdir(test_dir)
for flower in flowers:
    os.mkdir(os.path.join(train_dir, flower))
    os.mkdir(os.path.join(test_dir, flower))

data = list(zip(os.listdir(source_dir), flower_array))
np.random.seed(21)
np.random.shuffle(data)
# print(data)
train_size = int(len(data)*0.9)
for file, dir in data[:train_size]:
    source = os.path.join(source_dir, file)
    dest = os.path.join(train_dir, dir, file)
    print(f"copy file {source} -> {dest}")
    shutil.copy(source, dest)
for file, dir in data[train_size:]:
    source = os.path.join(source_dir, file)
    dest = os.path.join(test_dir, dir, file)
    print(f"copy file {source} -> {dest}")
    shutil.copy(source, dest)






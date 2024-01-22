import os.path
import random
import shutil

data_path = "./data"
train_path = "./train"
val_path = "./val"

if os.path.exists(train_path):
    shutil.rmtree(train_path)
if os.path.exists(val_path):
    shutil.rmtree(val_path)

os.makedirs(os.path.join(train_path, "images"))
os.makedirs(os.path.join(train_path, "labels"))
os.makedirs(os.path.join(val_path, "images"))
os.makedirs(os.path.join(val_path, "labels"))

files = [os.path.splitext(file)[0] for file in os.listdir(os.path.join(data_path, "images"))]
random.shuffle(files)
mid = int(len(files)*0.8)
for file in files[:mid]:
    source = os.path.join(data_path, "images", f"{file}.jpg")
    target = os.path.join(train_path, "images", f"{file}.jpg")
    shutil.copy(source, target)
    source = os.path.join(data_path, "labels", f"{file}.txt")
    target = os.path.join(train_path, "labels", f"{file}.txt")
    shutil.copy(source, target)

for file in files[mid:]:
    source = os.path.join(data_path, "images", f"{file}.jpg")
    target = os.path.join(val_path, "images", f"{file}.jpg")
    shutil.copy(source, target)
    source = os.path.join(data_path, "labels", f"{file}.txt")
    target = os.path.join(val_path, "labels", f"{file}.txt")
    shutil.copy(source, target)

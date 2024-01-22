# https://www.kaggle.com/datasets/andrewmvd/car-plate-detection?resource=download
import os.path
import random
import shutil

car_path = "./car"
train_path = "./car/train"
val_path = "./car/val"

if os.path.exists(train_path):
    shutil.rmtree(train_path)
if os.path.exists(val_path):
    shutil.rmtree(val_path)

os.makedirs(os.path.join(train_path, "images"))
os.makedirs(os.path.join(train_path, "labels"))
os.makedirs(os.path.join(val_path, "images"))
os.makedirs(os.path.join(val_path, "labels"))

files = [os.path.splitext(file)[0] for file in os.listdir(os.path.join(car_path, "images"))]
random.shuffle(files)

train_size = int(len(files)*0.8)
for file in files[:train_size]:
    source = os.path.join("./car/images", f"{file}.png")
    target = os.path.join(train_path, "images", f"{file}.png")
    shutil.copy(source, target)
    print(f"Copy {source} -> {target}")

    source = os.path.join("./car/labels", f"{file}.txt")
    target = os.path.join(train_path, "labels", f"{file}.txt")
    shutil.copy(source, target)
    print(f"Copy {source} -> {target}")

for file in files[train_size:]:
    source = os.path.join("./car/images", f"{file}.png")
    target = os.path.join(val_path, "images", f"{file}.png")
    shutil.copy(source, target)
    print(f"Copy {source} -> {target}")

    source = os.path.join("./car/labels", f"{file}.txt")
    target = os.path.join(val_path, "labels", f"{file}.txt")
    shutil.copy(source, target)
    print(f"Copy {source} -> {target}")
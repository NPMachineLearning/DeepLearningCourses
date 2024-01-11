import os
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential
from keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from keras.optimizers import Adam

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

flower_types = []
with open("label.txt") as file:
    for line in file:
        line = line.strip()
        cols = line.split()
        flower_types.append(cols[2])

train_path = "./train_images"
test_path = "./test_images"

train_images = []
train_labels = []
test_images = []
test_labels = []


for flower in flower_types:
    for file in os.listdir(os.path.join(train_path, flower)):
        file_path = os.path.join(train_path, flower, file)
        img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = img[:,:,::-1]
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
        img = preprocess_input(img)
        train_images.append(img)
        train_labels.append(flower)
train_images = np.array(train_images)

for flower in flower_types:
    for file in os.listdir(os.path.join(test_path, flower)):
        file_path = os.path.join(test_path, flower, file)
        img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = img[:,:,::-1]
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
        img = preprocess_input(img)
        test_images.append(img)
        test_labels.append(flower)
test_images = np.array(test_images)

train_onehot = np.zeros([len(train_labels), len(flower_types)])
test_onehot = np.zeros([len(test_labels), len(flower_types)])

for i in range(len(train_onehot)):
    train_onehot[i][flower_types.index(train_labels[i])] = 1

for i in range(len(test_onehot)):
    test_onehot[i][flower_types.index(test_labels[i])] = 1

#建模
model_base=VGG19(weights="imagenet",include_top=False,input_shape=(224,224,3))
for layer in model_base.layers:
    layer.trainable=False
model=Sequential()
model.add(model_base)
model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(64, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.2))#丟掉 20% 的資料，目對是為了防止過度擬合()
model.add(Dense(17, activation="softmax"))
model.compile(optimizer = Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images,
                    train_onehot,
                    epochs=50,
                    batch_size=32,
                    validation_data=(test_images, test_onehot),
                    verbose=1)


if os.path.exists("flower_model"):
    shutil.rmtree("flower_model")
model.save("flower_model")

plt.figure(figsize=(12, 7))
plt.plot(history.history["loss"], label="Train loss")
plt.plot(history.history["val_loss"], label="Val loss")
plt.plot(history.history["accuracy"], label="Train accuracy")
plt.plot(history.history["val_accuracy"], label="Val accuracy")
plt.show()



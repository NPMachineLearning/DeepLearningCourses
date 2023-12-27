# dataset http://download.tensorflow.org/example_images/flower_photos.tgz
import os
import shutil

import cv2
import numpy
import numpy as np
import pylab as plt
import keras
from keras import Sequential
from keras.applications import VGG19
from keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from keras.optimizers import Adam

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

if os.path.exists("flower_model"):
    shutil.rmtree("flower_model")

path = "./flower_photos"
images = []
labels = []

for flower in os.listdir(path):
    for file in os.listdir(os.path.join(path, flower)):
        file_path = os.path.join(path, flower, file)
        print(f"Load image: {file_path}")
        img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224,224), interpolation=cv2.INTER_LINEAR)
        images.append(img)
        labels.append(flower)

# random shuffle
upset = list(zip(images, labels))
np.random.seed(32)
np.random.shuffle(upset)
images, labels = zip(*upset)
images, labels = np.array(images), np.array(labels)

# split data
train = int(len(images)*0.9)
train_imgs, test_imgs = images[:train], images[train:]
train_labels, test_labels = labels[:train], labels[train:]

print(len(train_imgs), len(train_labels), len(test_imgs), len(test_labels))

# onehot encoding
train_onehot = np.zeros([len(train_labels), 5])
test_onehot = np.zeros([len(test_labels), 5])

kind = {n:i for i, n in enumerate(os.listdir(path))}
print(kind)

for i in range(len(train_onehot)):
    train_onehot[i][kind[train_labels[i]]] = 1
for i in range(len(test_onehot)):
    test_onehot[i][kind[test_labels[i]]] = 1

model_base = VGG19(weights="imagenet", include_top=False, input_shape=(224,224, 3))
for layer in model_base.layers:
    layer.trainable = False

model = Sequential()
model.add(model_base)
model.add(GlobalAveragePooling2D())
model.add(Dense(256, "relu"))
model.add(BatchNormalization())
model.add(Dense(64, "relu"))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(5, "softmax"))

model.compile(optimizer=Adam(learning_rate=1e-3),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(x=train_imgs,
                    y=train_onehot,
                    epochs=50,
                    batch_size=64,
                    validation_data=(test_imgs, test_onehot))

model.save("flower_model")

plt.plot(history.history["accuracy"], label="Train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.plot(history.history["loss"], label="Train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.legend()
plt.show()
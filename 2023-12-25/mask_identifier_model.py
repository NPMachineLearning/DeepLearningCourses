# dataset: https://github.com/prajnasb/observations

import os
import shutil

import cv2
import numpy as np
from keras.applications import VGG19
from keras import layers, Model
from keras.applications.vgg19 import preprocess_input
from keras.utils import to_categorical

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

if os.path.exists("mask_19"):
    shutil.rmtree("mask_19")

base_model = VGG19(weights="imagenet", include_top=False)

for layer in base_model.layers:
    print(layer)
    layer.trainable = False

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(units=256, activation="relu")(x)
x = layers.Dense(units=64, activation="relu")(x)
x = layers.Dense(units=2, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=x)

model.compile(loss="binary_crossentropy", optimizer="adam")

path = "./data/without_mask"
files_nomask = [os.path.join(path, f) for f in os.listdir(path)]
path = "./data/with_mask"
files_mask = [os.path.join(path, f) for f in os.listdir(path)]

X = []
for files in [files_nomask, files_mask]:
    for file in files:
        print(f"Load image: {file}")
        img = cv2.imdecode(np.fromfile(file, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224,224))
        img = img[:,:,::-1]
        X.append(img)

X = np.array(X)
X = preprocess_input(X)
y = np.zeros([len(files_nomask)+len(files_mask)])
y[len(files_nomask):] = 1
y = to_categorical(y, num_classes=2)

model.fit(X, y, epochs=20, validation_split=0.2, verbose=1)
model.save("mask_19")
import os

import cv2
import keras.models
import matplotlib.pyplot as plt
import numpy as np
from keras.applications.vgg19 import preprocess_input

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

model = keras.models.load_model("flower_model")

image_dir = "./images"

flower_types = []
with open("label.txt") as file:
    for line in file:
        line = line.strip()
        cols = line.split()
        flower_types.append(cols[2])

plt.tight_layout()
for i, file in enumerate(os.listdir(image_dir)):
    file_path = os.path.join(image_dir, file)
    img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    img = img[:,:,::-1]
    x = cv2.resize(img, (224,224), interpolation=cv2.INTER_LINEAR)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    result = model.predict(x)
    label_index = np.argmax(result, axis=1)[0]
    prob = np.round(result[0][label_index] * 100.0, 2)
    flower_name = flower_types[label_index]
    plt.subplot(5,3,i+1)
    plt.tight_layout()
    plt.imshow(img)
    plt.title(label=f"{flower_name} | {prob}%",
              c="r" if prob < 50.0 else "g")
    plt.axis(False)
plt.show()
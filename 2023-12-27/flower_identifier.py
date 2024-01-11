import os

import cv2
import keras.models
import matplotlib.pyplot as plt
import numpy as np
from keras.applications.vgg19 import preprocess_input

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

model = keras.models.load_model("flower_model")
path = "./images"
data_path = "./flower_photos"
kind = {i:n for i, n in enumerate(os.listdir(data_path))}
print(kind)
plt.figure(figsize=(12,10))
for i, file in enumerate(os.listdir(path)):
    file_path = os.path.join(path, file)
    img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    img = img[:,:,::-1]
    img = cv2.resize(img, (224,224), interpolation=cv2.INTER_LINEAR)
    x = preprocess_input(img)
    x = np.expand_dims(img, axis=0)
    result = model.predict(x)
    result = np.argmax(result, axis=1)[0]
    plt.subplot(5,3,i+1)
    plt.imshow(img)
    plt.title(kind[result])
    plt.axis(False)
plt.show()

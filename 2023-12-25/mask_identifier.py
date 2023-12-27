# dataset https://mahaljsp.ddns.net/files/mask_images.zip
import os
import cv2
import numpy as np
import keras
from keras.applications.vgg19 import preprocess_input
import pylab as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
path = "./mask_images/images"

model = keras.models.load_model("mask_19")

plt.figure(figsize=(12, 10))
for i, file in enumerate(os.listdir(path)):
    file_path = os.path.join(path, file)
    img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    img = img[:,:,::-1]
    img = cv2.resize(img, (224,224), interpolation=cv2.INTER_LINEAR)
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    result = model.predict(x)
    result = np.squeeze(result, axis=0)
    result = np.argmax(result, axis=0)
    text = ""
    if result > 0:
        text = "mask"
    else:
        text = "no mask"
    plt.subplot(3,6,i+1)
    plt.imshow(img)
    plt.title(text)
    plt.axis(False)
plt.show()
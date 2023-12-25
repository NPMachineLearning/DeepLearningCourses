import os

from keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.applications.vgg19 import decode_predictions
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import numpy as np
import pylab as plt
import cv2

os.environ["TF_CPP_LOG_MIN_LEVEL"] = "2"

vgg19_model = VGG19(weights="imagenet", include_top=True)

# load and preprocessing image
img = cv2.imdecode(np.fromfile("dog3.jpg", dtype=np.uint8), cv2.IMREAD_COLOR)
X = cv2.resize(img, (224, 224))
X = X[:,:,::-1]
X = np.expand_dims(X, 0)
X = preprocess_input(X)
print(X.shape)

out = vgg19_model.predict(X)
result = decode_predictions(out, 3)
name = result[0][0][1]
prob = result[0][0][2]
print(f"Name: {name} | Probability: {prob*100.0:.2f}%")

pil_img = Image.fromarray(img[:,:,::-1].copy())
txt = f"Name: {name} | Probability: {prob*100.0:.2f}%"
ImageDraw.Draw(pil_img).text((5,5), text=txt, font=ImageFont.truetype("arial.ttf", size=15))
plt.imshow(pil_img)
plt.show()

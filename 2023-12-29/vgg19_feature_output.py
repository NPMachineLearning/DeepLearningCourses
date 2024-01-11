import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input
import tensorflow as tf

base_model = VGG19(include_top=False)
outputs_dict = dict([layer.name, layer.output] for layer in base_model.layers)
model = keras.Model(inputs=base_model.input, outputs=outputs_dict)

img_path = "image.jpg"
img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
img = cv2.resize(img, (224,224), interpolation=cv2.INTER_LINEAR)
img = img[:,:,::-1]
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)
outputs = model(img)
for layer_name in outputs:
    print(layer_name, outputs[layer_name].shape)

layer = outputs["block5_conv4"]
r, h, w, c = layer.shape
layer = tf.reshape(layer, (h,w,c))

plt.figure(figsize=(12,7))
for i in range(64):
    plt.subplot(8,8, i+1)
    plt.imshow(layer[:,:,i], cmap="gray")
    plt.axis(False)
plt.show()

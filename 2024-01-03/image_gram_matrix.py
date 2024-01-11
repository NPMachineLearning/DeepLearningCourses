import numpy as np
import cv2

path = "./vango.jpg"

image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
h, w, _ = image.shape
height = 100
width = round(w*height/h)
image = cv2.resize(image, (width, height))
image = image[:,:,::-1].copy()
print(image.shape)
image = np.transpose(image, axes=(2,0,1))
print(image.shape)
a = image.reshape([3, -1]).astype(np.int32)
print(a.shape)
b = a.T
print(b.shape)
c = np.outer(a, b)
print(c.shape)
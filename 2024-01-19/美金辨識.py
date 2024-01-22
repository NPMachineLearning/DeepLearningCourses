import os.path
import platform

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from ultralytics import YOLO


def text(img, txt, xy=(0,0), color=(0,0,0), size=12):
    pil = Image.fromarray(img)
    s = platform.system()
    if s == "Linux":
        font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", size)
    elif s == "darwin":
        pass
    else:
        font = ImageFont.truetype("simsun.ttc", size)

    draw = ImageDraw.Draw(pil)
    draw.text(xy, txt, font=font, fill=color)
    return np.array(pil)

model = YOLO("./best.pt")
data_path = "./dollar/test/images"
files=['IMG_1909_jpg.rf.f3843edb1f3c3fb533faba5db0921569.jpg',
       'IMG_1911_jpg.rf.39f98a04fc8e36c9c6b8b1fd66a55ead.jpg',
       'IMG_1916_jpg.rf.5e68fc1f7707c0083b82a297b4d95796.jpg',
       'IMG_1919_jpg.rf.aa2329edb7f74fd40440e4a783e58b3c.jpg']
for i, f in enumerate(files):
    path = os.path.join(data_path, f)
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    img = img[:,:,::-1].copy()
    results = model.predict(img, save=False)
    boxes = results[0].boxes.xyxy

    names = [results[0].names[int(i.cpu().numpy())] for i in results[0].boxes.cls]
    for box, name in zip(boxes, names):
        box = box.cpu().numpy()
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        img = cv2.rectangle(img, (x1,y1), (x2,y2), (0, 255, 0), 1)
        img = text(img, name, (x1,y1-20), (0,0,255), 16)
    plt.subplot(2,2,i+1)
    plt.imshow(img)
plt.show()
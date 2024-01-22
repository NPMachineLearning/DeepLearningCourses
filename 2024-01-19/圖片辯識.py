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

model = YOLO("yolov8x.pt")
img = cv2.imdecode(np.fromfile("test.jpg", dtype=np.uint8), cv2.IMREAD_COLOR)
img = img[:,:,::-1].copy()
results = model(img)
print(results[0].boxes.cls)
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

plt.imshow(img)
plt.show()
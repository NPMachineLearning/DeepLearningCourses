import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import platform

from ultralytics import YOLO


def text(img, txt, xy=(0,0), color=(0,0,0), size=12):
    pil=Image.fromarray(img)#cv2 轉 Pillow 格式
    s=platform.system()
    if s == "Linux":
        font=ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", size)
    elif s == "Darwin": #Mac 系統
        font = ImageFont.truetype("自已查", size)
    else:#Windows 系統
        font = ImageFont.truetype("simsun.ttc", size)
    draw=ImageDraw.Draw(pil)
    draw.text(xy, txt, font=font, fill=color)
    return np.array(pil)#Pillow 轉 cv2 格式

model = YOLO("fruits.pt")
img = cv2.imdecode(np.fromfile("./val/images/pitaya_00011.jpg", dtype=np.uint8), cv2.IMREAD_COLOR)
img = img[:,:,::-1].copy()
results = model.predict(img)
boxes = results[0].boxes.xyxy
names =[results[0].names[int(idx.cpu().numpy())] for idx in results[0].boxes.cls]
for box, name in zip(boxes, names):
    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[2])
    y2 = int(box[3])
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 5)
    img = text(img, name, (x1, y1-50), (255,0,0), 50)
plt.imshow(img)
plt.axis(False)
plt.show()
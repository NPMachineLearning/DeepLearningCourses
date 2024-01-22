import platform

import cv2
from PIL import Image, ImageFont, ImageDraw
import numpy as np
from matplotlib import pyplot as plt
from pytesseract import pytesseract
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

model = YOLO("./car/car.pt")
plt.figure(figsize=(10, 7))
for i, file in enumerate(["car1.jpg", "car2.jpg", "car3.jpg", "car4.jpg"]):
    img = cv2.imdecode(np.fromfile(file, dtype=np.uint8), cv2.IMREAD_COLOR)
    img = img[:,:,::-1].copy()

    results = model.predict(img, save=False)
    boxes = results[0].boxes.xyxy
    for box in boxes:
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 5)
        tmp = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_RGB2GRAY)
        license = pytesseract.image_to_string(tmp, lang="eng", config="--psm 11")
        img = text(img, txt=license, xy=(x1, y1-20), color=(0, 255, 0), size=100)

    plt.subplot(2,2,i+1)
    plt.axis(False)
    plt.imshow(img)
plt.show()
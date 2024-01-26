import platform

import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from PyQt5.QtCore import QThread, pyqtSignal


class DetectThread(QThread):
    callback = pyqtSignal(object)
    def __init__(self, model, file, parent=None):
        super().__init__(parent)
        self.model = model
        self.file = file

    def text(self, img, txt, xy=(0, 0), color=(0, 0, 0), size=12):
        pil = Image.fromarray(img)
        s = platform.system()
        if s == "Linux":
            font = ImageFont.truetype('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', size)
        elif s == "Darwin":
            font = ImageFont.truetype('....', size)
        else:
            font = ImageFont.truetype('simsun.ttc', size)
        ImageDraw.Draw(pil).text(xy, txt, font=font, fill=color)
        return np.array(pil)
    def run(self):
        img = cv2.imdecode(np.fromfile(self.file, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = img[:,:,::-1].copy()
        results = self.model.predict(img)[0]
        names = [results.names[int(i.cpu().numpy())] for i in results.boxes.cls]
        boxes = results.boxes.xyxy
        for box, name in zip(boxes, names):
            box = box.cpu().numpy()
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 5)
            img = self.text(img, name, (x1, y1-20), (0,255,0), 20)
        self.callback.emit(img)
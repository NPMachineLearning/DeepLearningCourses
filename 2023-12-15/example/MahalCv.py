#pip install opencv-python
#Mahal sdk : software development kit(軟体開發套件)
import cv2
import numpy as np
class MahalCv():
    @staticmethod
    def read(filePath):#filePath允許中文檔名
        #IMREAD_UNCHANGED : 只能讀 jpg, bmp(RGB)
        #IMREAD_COLOR : 可能讀取 png(RGBA)，然後轉成 BGR, 將 A 去除
        # img =cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), cv2.IMREAD_COLOR)
        # return img
        return cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), cv2.IMREAD_COLOR)
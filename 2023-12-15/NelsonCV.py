import cv2
import numpy as np
class NelsonCV():
    @staticmethod
    def read(filePath, size=None):
        img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), cv2.IMREAD_COLOR)
        if size is not None:
            img = cv2.resize(img, size)
        return img
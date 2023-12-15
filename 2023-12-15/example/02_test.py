#from MahalCv.py檔 import MahalCv類別
from MahalCv import MahalCv as cv
import cv2
img=cv.read("老虎.jpg")
img=cv2.resize(img, (800,600), cv2.INTER_LINEAR)
cv2.imshow("tiger", img)
cv2.waitKey(0)
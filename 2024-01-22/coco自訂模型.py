# https://cocodataset.org/#download
import time
from ultralytics import YOLO

if __name__=='__main__':
    model = YOLO("yolov8n.pt")
    print("開始訓練 .........")
    t1=time.time()
    model.train(data="./coco/data.yaml", epochs=200, imgsz=640)
    t2=time.time()
    print(f'訓練花費時間 : {t2-t1}秒')
    path=model.export()
    print(f'模型匯出路徑 : {path}')
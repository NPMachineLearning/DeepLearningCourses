from time import time

from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    t1 = time()
    model.train(data="./data.yaml", epochs=200, imgsz=640)
    t2 = time()
    print(f"train time: {t2-t1} seconds")
    path = model.export()
    print(f"model save path: {path}")
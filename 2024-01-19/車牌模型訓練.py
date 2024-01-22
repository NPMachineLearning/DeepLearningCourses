import os.path
import shutil
import time

from ultralytics import YOLO

if __name__ == "__main__":
    # train_path = "./car/runs/detect/train"
    # if os.path.exists(train_path):
    #     shutil.rmtree(train_path)
    model = YOLO("yolov8l.pt")
    time1 = time.time()
    model.train(data="./car/data.yaml", epochs=150, imgsz=640)
    time2 = time.time()
    print(f"Train time: {time2 - time1} seconds")
    path = model.export()
    print(f"model saved path: {path}")
# commandline training
# yolo task=detect mode=train model=./yolov8n.pt data=./Dollar/data.yaml epochs=200 imgsz=640
import os.path
import shutil
import time

from ultralytics import YOLO

if __name__ == "__main__":
    train_path = "./runs/detect/train"
    if os.path.exists(train_path):
        shutil.rmtree(train_path)

    model = YOLO("yolov8n.pt")
    time1 = time.time()
    model.train(data="./dollar/data.yaml", epochs=200, imgsz=640)
    time2 = time.time()
    print(f"Train time: {time2 - time1} seconds")

# https://mahaljsp.ddns.net/files/yolov8/yolov8_coco.zip

from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("./runs/detect/train/weights/last.pt")
    results = model.train(resume=True)
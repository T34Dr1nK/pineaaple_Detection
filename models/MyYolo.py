from ultralytics import YOLO
import shutil
import os

model = YOLO("yolov12")

run_name = "pineappleV12"
model.train(
    data="something.yaml", epochs=50, imgsz=640, batch=16, device=0, name=run_name
)

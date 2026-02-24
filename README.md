# Pineapple Detection Benchmark Framework

A modular object detection training and benchmarking framework supporting:

- YOLOv11 (Ultralytics)
- YOLOv12 (Ultralytics)
- Faster R-CNN (Torchvision)

This project is designed for research comparison, experimentation, and clean architecture implementation.

---

## Project Structure

```
pineapple_detection/
│
├── train.py
├── inference.py
│
├── configs/
│ ├── yolov11.yaml
│ ├── yolov12.yaml
│ └── fasterrcnn.yaml
│
├── models/
│ ├── base_trainer.py
│ ├── yolo_trainer.py
│ └── fasterrcnn_trainer.py
│
├── data/
│ ├── dataset_validator.py
│ └── coco_dataset.py
│
├── utils/
│ ├── seed.py
│ └── logger.py
│
├── datasets/
│ ├── yolo/
│ └── coco/
│
├── test_images/
└── runs/
```

---

## Installation

```bash
pip install -r requirements.txt
```

If using GPU, install correct PyTorch version from:

https://pytorch.org/get-started/locally/

---

## Dataset Setup

### YOLO Format

Place dataset inside:\
`datasets/yolo/`

Structure:\
`train/images`\
`train/labels`\
`valid/images`\
`data.yaml`\

Make sure data.yaml paths are relative:

```yaml
train: train/images
val: valid/images
test: test/images
nc: 3
names: ["ripe", "unripe", "damaged"]
```

---

### COCO Format (Faster R-CNN)

Place dataset inside:\
`datasets/coco/`

Example:\
`train/`\
`valid/`\
`annotations.json`

---

## Training

### YOLOv12

```bash
python train.py --model yolo12 --config configs/yolo12.yaml
```

### YOLOv11

```bash
python train.py --model yolo11 --config configs/yolo11.yaml
```

### Faster R-CNN

```bash
python train.py --model fasterrcnn --config configs/fasterrcnn.yaml
```

### Outputs will be saved inside:

`runs/<model_name>/`

---

## Inference

### YOLO

```bash
python inference.py \
--model yolo12 \
--weights runs/yolo12/<your_run_name>/weights/best.pt \
--source test_images/test.jpg \
--runname inference_run #or whatever you want
```

### Output:

`runs/inference/inference_run/`

---

### Faster R-CNN

```bash
python inference.py \
--model fasterrcnn \
--weights runs/fasterrcnn/model.pth \
--source test_images/test.jpg \
--num_classes 4
```

---

## Author

- T34_Dr1nK

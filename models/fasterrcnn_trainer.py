import os
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from .base_trainer import BaseTrainer
from data.dataset_validator import validate_coco_json


class FasterRCNNTrainer(BaseTrainer):
    def train(self):
        annotation = self.config["annotation"]
        validate_coco_json(annotation)

        num_classes = self.config["num_classes"]

        model = fasterrcnn_resnet50_fpn(num_classes=num_classes)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print("✅ Faster R-CNN model initialized")
        print("⚠ You must implement DataLoader + training loop in coco_dataset.py")

        os.makedirs("runs/fasterrcnn", exist_ok=True)

import torch
from ultralytics import YOLO
from .base_trainer import BaseTrainer
from data.dataset_validator import validate_yolo_yaml


class YOLOTrainer(BaseTrainer):
    def __init__(self, model_name, config):
        super().__init__(config)
        self.model_name = model_name + "n"

    def train(self):
        data_path = self.config["data"]
        validate_yolo_yaml(data_path)

        model = YOLO(f"{self.model_name}.pt")

        model.train(
            data=data_path,
            epochs=self.config.get("epochs", 100),
            imgsz=self.config.get("imgsz", 640),
            batch=self.config.get("batch", 16),
            device=0 if torch.cuda.is_available() else "cpu",
            project=f"runs/{self.model_name}",
            name=self.config.get("experiment_name", "exp"),
            exist_ok=True,
        )

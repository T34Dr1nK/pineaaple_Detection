import os
import json
import torch
from PIL import Image


class COCODataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation):
        self.root = root
        self.annotation = annotation

        with open(annotation) as f:
            coco = json.load(f)

        self.images = coco["images"]
        self.annotations = coco["annotations"]

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.root, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []

        for ann in self.annotations:
            if ann["image_id"] == img_info["id"]:
                x, y, w, h = ann["bbox"]
                boxes.append([x, y, x + w, y + h])
                labels.append(ann["category_id"])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }

        return img, target

    def __len__(self):
        return len(self.images)

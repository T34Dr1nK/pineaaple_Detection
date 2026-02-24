import os
import yaml


def validate_yolo_yaml(path):
    print(path)
    if not os.path.exists(path):
        raise FileNotFoundError("data.yaml not found")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    required = ["train", "val", "nc", "names"]
    for key in required:
        if key not in data:
            raise ValueError(f"{key} missing in YAML")

    print("✅ YOLO dataset format valid")


def validate_coco_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError("COCO annotation file not found")

    print("✅ COCO annotation file exists")

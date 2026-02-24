import argparse
import os
import torch

from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
import torchvision.transforms as T


def yolo_inference(weights, source, run_name):
    print(f"✅ Loading YOLO model from {weights}")
    model = YOLO(weights)

    results = model.predict(
        source=source, save=True, project="runs/inference", name=run_name, exist_ok=True
    )

    print("✅ Inference completed")
    return results


def fasterrcnn_inference(weights, source, num_classes, run_name):
    print(f"✅ Loading Faster R-CNN model from {weights}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = fasterrcnn_resnet50_fpn(num_classes=num_classes)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device)
    model.eval()

    image = Image.open(source).convert("RGB")
    transform = T.ToTensor()
    image_tensor = transform(image).to(device)

    with torch.no_grad():
        outputs = model([image_tensor])

    print("✅ Inference completed")
    print(outputs)

    os.makedirs("runs/inference/fasterrcnn", exist_ok=True)
    return outputs


def main(args):
    if args.model in ["yolov11", "yolov12"]:
        yolo_inference(weights=args.weights, source=args.source, run_name=args.runname)

    elif args.model == "fasterrcnn":
        fasterrcnn_inference(
            weights=args.weights,
            source=args.source,
            num_classes=args.num_classes,
            run_name=args.runname,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model", required=True, choices=["yolov11", "yolov12", "fasterrcnn"]
    )

    parser.add_argument("--weights", required=True, help="Path to trained weights")

    parser.add_argument("--source", required=True, help="Image or folder for inference")

    parser.add_argument("--runname", default="inference_run")

    parser.add_argument(
        "--num_classes", type=int, default=4, help="Required for FasterRCNN"
    )

    args = parser.parse_args()
    main(args)

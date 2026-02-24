import argparse
import yaml

from models.yolo_trainer import YOLOTrainer
from models.fasterrcnn_trainer import FasterRCNNTrainer
from utils.seed import set_seed


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, choices=["yolov11", "yolov12", "fasterrcnn"]
    )
    parser.add_argument("--config", required=True)

    args = parser.parse_args()
    config = load_config(args.config)

    set_seed(config.get("seed", 42))

    if args.model in ["yolov11", "yolov12"]:
        trainer = YOLOTrainer(args.model, config)
    else:
        trainer = FasterRCNNTrainer(config)

    trainer.train()


if __name__ == "__main__":
    main()

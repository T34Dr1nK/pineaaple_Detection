import logging
import os


def setup_logger(log_dir, name="train.log"):
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, name)

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    return logging.getLogger()

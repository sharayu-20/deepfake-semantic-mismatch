"""Lightweight per-epoch metric logger used by scripts/train.py."""

import os
from datetime import datetime


class EpochLogger:
    def __init__(self, log_dir: str, run_name: str):
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.path = os.path.join(log_dir, f"{run_name}_{ts}.txt")

    def log(self, **fields):
        line = " ".join(f"{k}: {v}" for k, v in fields.items())
        with open(self.path, "a+") as f:
            f.write(line + "\n")
        print(line)

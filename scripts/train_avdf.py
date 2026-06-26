#!/usr/bin/env python3
"""
Training loop for AVDF_Multiclass_Semantic (PyTorch Lightning), five-class
RARV-SMM setting. Separate from train.py since AVDF follows the AV-HuBERT /
Lightning k-fold conventions of the underlying AVDF/MRDF backbone, rather than
the plain training loop shared by FGMDF and FGI.

Usage:
    python scripts/train_avdf.py --data_root data/combined_5class \
        --scores_json semantic_scores_v1.json --save_name avdf_sem_v1
"""

import argparse
import os
import sys

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.avdf_semantic import AVDF_Multiclass_Semantic
from utils.dataset_avdf import FakeavcelebSemanticDataModule  # see utils/dataset_avdf.py


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--scores_json", required=True)
    parser.add_argument("--save_name", default="avdf_sem_v1")
    parser.add_argument("--outputs", default="summary/avdf")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--min_epochs", type=int, default=30)
    # NOTE: min_epochs == max_epochs by default, so EarlyStopping can never actually
    # fire here regardless of patience (Lightning won't stop before min_epochs).
    # If you raise --max_epochs above --min_epochs, patience=0 would stop training
    # after the very first non-improving epoch -- raise this together with max_epochs.
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    args = parser.parse_args()

    seed_everything(42)

    model = AVDF_Multiclass_Semantic(weight_decay=args.weight_decay, learning_rate=args.learning_rate)
    dm = FakeavcelebSemanticDataModule(root=args.data_root, scores_json=args.scores_json,
                                        train_fold="train_1.txt", test_fold="test_1.txt",
                                        batch_size=args.batch_size, num_workers=args.num_workers)

    trainer = Trainer(
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        callbacks=[
            ModelCheckpoint(dirpath=f"{args.outputs}/ckpts", save_top_k=3,
                            filename=args.save_name + "_{epoch}-{val_re:.4f}", monitor="val_re", mode="max"),
            EarlyStopping(monitor="val_re", patience=args.patience, mode="max"),
        ],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
    )
    trainer.fit(model, dm)
    result = trainer.test(model, dm.test_dataloader(), ckpt_path="best")
    print(result)


if __name__ == "__main__":
    main()

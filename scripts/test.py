#!/usr/bin/env python3
"""
Evaluation/testing loop — five-class semantic-reinforced detection (FGMDF/FGI).

Loads a checkpoint produced by train.py, runs it over a test set, and reports
per-class precision/recall/F1, macro/weighted AUC-ROC, and the confusion matrix
(counts and percentages), matching the metrics reported in the paper.

Usage:
    python scripts/test.py --model fgmdf --checkpoint summary/weight/5class_fgmdf_v1_imagebind/12.pth \
        --test_txt data_path/test_path_5class_v1.txt --scores_json semantic_scores_v1.json \
        --output_dir results/fgmdf_v1
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
                              confusion_matrix, precision_recall_fscore_support, roc_auc_score)
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.dataset import Multimodal_dataset_semantic, collate_fn_semantic

NUM_CLASSES = 5
CLASS_NAMES = ["RARV", "RAFV", "FARV", "FAFV", "RARV-SMM"]


def build_model(model_name, num_classes):
    if model_name == "fgmdf":
        from models.fgmdf_semantic import GAT_video_audio_semantic_v3
        return GAT_video_audio_semantic_v3(num_classes=num_classes, audio_nodes=4)
    if model_name == "fgi":
        from models.fgi_semantic import My_Network_Semantic
        return My_Network_Semantic(num_classes=num_classes)
    raise ValueError(f"Unsupported model: {model_name}")


def load_checkpoint(model, checkpoint_path, device):
    state_dict = torch.load(checkpoint_path, map_location=device)
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    preds, labels, probs = [], [], []
    for _, video, audio, semantic_score, total_label, *_ in tqdm(loader, desc="Evaluating"):
        video, audio, semantic_score = video.to(device), audio.to(device), semantic_score.to(device)
        total_output, *_ = model(video, audio, semantic_score)
        probs.append(torch.softmax(total_output, dim=-1).cpu().numpy())
        preds.append(torch.argmax(total_output, dim=1).cpu().numpy())
        labels.append(total_label.numpy())
    return np.concatenate(preds), np.concatenate(labels), np.concatenate(probs)


def compute_metrics(y_true, y_pred, y_probs):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(NUM_CLASSES), average=None, zero_division=0)
    per_class = {CLASS_NAMES[i]: {"precision": float(precision[i]), "recall": float(recall[i]),
                                   "f1": float(f1[i]), "support": int(support[i])}
                 for i in range(NUM_CLASSES)}
    auc = roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")
    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
    return {"accuracy": accuracy_score(y_true, y_pred), "macro_auc": float(auc),
            "per_class": per_class, "confusion_matrix": cm.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["fgmdf", "fgi"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test_txt", required=True)
    parser.add_argument("--scores_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--num_frame", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gpu", type=str, default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    test_dataset = Multimodal_dataset_semantic(args.image_size, "test", args.test_txt,
                                                num_frame=args.num_frame, scores_json=args.scores_json)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                               collate_fn=collate_fn_semantic)

    model = build_model(args.model, NUM_CLASSES).to(device)
    model = load_checkpoint(model, args.checkpoint, device)
    model.eval()

    preds, labels, probs = evaluate(model, test_loader, device)
    metrics = compute_metrics(labels, preds, probs)

    print(f"Accuracy: {metrics['accuracy']:.4f}  Macro AUC: {metrics['macro_auc']:.4f}")
    print(classification_report(labels, preds, target_names=CLASS_NAMES))

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics -> {args.output_dir}/metrics.json")


if __name__ == "__main__":
    main()

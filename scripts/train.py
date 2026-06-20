#!/usr/bin/env python3
"""
Main training loop — five-class (RARV, RAFV, FARV, FAFV, RARV-SMM) detection with
ImageBind semantic reinforcement, for the FGMDF and FGI backbones (shared
forward(video, audio, semantic_score) signature). For AVDF, see train_avdf.py
(PyTorch Lightning).

Pre-requisite: run compute_semantic_scores.py to generate the scores JSON for
the chosen RARV-SMM variant.

Usage:
    python scripts/train.py --model fgmdf --variant v1 \
        --train_txt data_path/train_path_5class_v1.txt \
        --test_txt  data_path/test_path_5class_v1.txt \
        --scores_json semantic_scores_v1.json \
        --weights_dir summary/weight
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.optim import lr_scheduler
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.dataset import Multimodal_dataset_semantic, collate_fn_semantic
from utils.logger import EpochLogger

NUM_CLASSES = 5
CLASS_NAMES = ["RARV", "RAFV", "FARV", "FAFV", "RARV-SMM"]


def build_model(model_name, num_classes):
    if model_name == "fgmdf":
        from models.fgmdf_semantic import GAT_video_audio_semantic_v3
        return GAT_video_audio_semantic_v3(num_classes=num_classes, audio_nodes=4)
    if model_name == "fgi":
        from models.fgi_semantic import My_Network_Semantic
        return My_Network_Semantic(num_classes=num_classes)
    raise ValueError(f"Unsupported model for this loop: {model_name} (use train_avdf.py for avdf)")


def run_epoch(model, loader, optimizer, criterion, binary_criterion, device, train: bool):
    model.train() if train else model.eval()
    total_loss, total_correct, n = 0.0, 0.0, 0
    all_probs, all_labels, all_preds = [], [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for name, video, audio, semantic_score, total_label, video_label, audio_label in tqdm(loader):
            video, audio = video.to(device), audio.to(device)
            semantic_score = semantic_score.to(device)
            total_label, video_label, audio_label = (
                total_label.to(device), video_label.to(device), audio_label.to(device))

            if train:
                optimizer.zero_grad()

            total_output, video_output, audio_output, _ = model(video, audio, semantic_score)
            loss = (criterion(total_output, total_label)
                    + binary_criterion(video_output, video_label)
                    + binary_criterion(audio_output, audio_label))

            if train:
                loss.backward()
                optimizer.step()

            preds = torch.argmax(total_output, dim=1)
            total_loss += loss.item()
            total_correct += torch.sum(preds == total_label).item()
            n += total_label.size(0)

            all_probs.append(torch.softmax(total_output, dim=-1).detach().cpu())
            all_labels.append(total_label.cpu())
            all_preds.append(preds.cpu())

    probs, labels, preds = torch.cat(all_probs), torch.cat(all_labels), torch.cat(all_preds)
    auc = roc_auc_score(F.one_hot(labels, NUM_CLASSES).numpy(), probs.numpy())
    cm = confusion_matrix(labels.numpy(), preds.numpy())
    return total_loss / n, total_correct / n, auc, cm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["fgmdf", "fgi"], required=True)
    parser.add_argument("--variant", choices=["v1", "v2", "v3"], required=True)
    parser.add_argument("--train_txt", required=True)
    parser.add_argument("--test_txt", required=True)
    parser.add_argument("--scores_json", required=True)
    parser.add_argument("--weights_dir", default="summary/weight")
    parser.add_argument("--log_dir", default="summary/result")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--num_frame", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1.1e-3)
    parser.add_argument("--gpu", type=str, default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_name = f"5class_{args.model}_{args.variant}_imagebind"
    weights_dir = os.path.join(args.weights_dir, run_name)
    os.makedirs(weights_dir, exist_ok=True)
    logger = EpochLogger(args.log_dir, run_name)

    train_dataset = Multimodal_dataset_semantic(args.image_size, "train", args.train_txt,
                                                 num_frame=args.num_frame, scores_json=args.scores_json)
    test_dataset = Multimodal_dataset_semantic(args.image_size, "test", args.test_txt,
                                                num_frame=args.num_frame, scores_json=args.scores_json)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                num_workers=4, collate_fn=collate_fn_semantic, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                               num_workers=4, collate_fn=collate_fn_semantic, pin_memory=True)

    model = build_model(args.model, NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    binary_criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    best_acc = 0.0
    for epoch in range(args.epochs):
        train_loss, train_acc, _, _ = run_epoch(model, train_loader, optimizer, criterion, binary_criterion, device, train=True)
        test_loss, test_acc, test_auc, test_cm = run_epoch(model, test_loader, optimizer, criterion, binary_criterion, device, train=False)
        scheduler.step()

        logger.log(epoch=epoch + 1, train_loss=f"{train_loss:.4f}", train_acc=f"{train_acc:.4f}",
                   test_loss=f"{test_loss:.4f}", test_acc=f"{test_acc:.4f}", test_auc=f"{test_auc:.4f}")
        print("confusion_matrix:\n", test_cm)

        if test_acc >= best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(weights_dir, f"{epoch + 1}.pth"))

    print(f"Best test accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()

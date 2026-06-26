#!/usr/bin/env python3
"""
Pre-compute ImageBind audio-visual cosine similarity scores for all samples.

Run once before training/evaluation — results are saved to a JSON file keyed by
frame directory path, the same key returned as `name` by
utils.dataset.Multimodal_dataset_semantic.__getitem__.

Usage:
    python scripts/compute_semantic_scores.py \
        --txt_files data_path/train_path_5class_v1.txt data_path/test_path_5class_v1.txt \
        --output    semantic_scores_v1.json
"""

import argparse
import glob
import json
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.semantic_scorer import ImageBindSemanticScorer


def parse_txt(txt_path):
    """Each line: <frame_path> <label> <start_second>. Resolve to (frame_dir, audio_path)."""
    entries = []
    with open(txt_path, "r") as fh:
        for line in fh:
            # "<frame_path> <label> <start_second>" -> drop the trailing two
            # whitespace-separated fields. (Don't slice off a fixed number of
            # characters: that breaks the moment label or start_second hits 2 digits.)
            img_path = line.rstrip().rsplit(maxsplit=2)[0]
            frame_dir = os.path.split(img_path)[0]

            audio_files = glob.glob(os.path.join(frame_dir, "16k_*.wav"))
            if audio_files:
                aud_path = audio_files[0]
            else:
                parent_audio = glob.glob(os.path.join(os.path.dirname(frame_dir), "16k_*.wav"))
                if parent_audio:
                    aud_path = parent_audio[0]
                else:
                    frame_number = os.path.splitext(os.path.basename(img_path))[0]
                    aud_path = os.path.join(frame_dir, f"16k_{frame_number}.wav")
            entries.append((frame_dir, aud_path))
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt_files", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--save_every", type=int, default=500)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading ImageBind on {device} ...")
    scorer = ImageBindSemanticScorer(device=device)

    all_entries = {}
    for txt_path in args.txt_files:
        for frame_dir, aud_path in parse_txt(txt_path):
            all_entries.setdefault(frame_dir, aud_path)
    print(f"Total unique samples: {len(all_entries)}")

    scores = {}
    if os.path.exists(args.output):
        with open(args.output) as f:
            scores = json.load(f)
        print(f"Resuming — {len(scores)} scores already computed.")

    remaining = [(fd, ap) for fd, ap in all_entries.items() if fd not in scores]
    for i, (frame_dir, aud_path) in enumerate(tqdm(remaining, desc="Computing scores")):
        try:
            scores[frame_dir] = scorer.score_directory(frame_dir, aud_path)
        except Exception as e:
            print(f"\nWarning: failed for {frame_dir}: {e}")
            scores[frame_dir] = 0.5

        if (i + 1) % args.save_every == 0:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(scores, f)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"Done. {len(scores)} scores saved to {args.output}")


if __name__ == "__main__":
    main()

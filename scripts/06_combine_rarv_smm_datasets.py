#!/usr/bin/env python3
"""
Phase 6 — Combine RARV-SMM V1, V2, V3 clips into a unified train/test split.

Samples a fixed number of clips from each variant's metadata (default: roughly
1/3 each), copies and renames the videos into one output directory, and writes
a unified metadata CSV with a normalized schema across variants.

Usage:
    python scripts/06_combine_rarv_smm_datasets.py \
        --v1-dir output_clips        --v1-meta output_clips/rarv_smm_v1_metadata.csv \
        --v2-dir output_clips_v2     --v2-meta output_clips_v2/rarv_smm_v2_metadata.csv \
        --v3-dir output_clips_v3     --v3-meta output_clips_v3/rarv_smm_v3_metadata.csv \
        --output-dir output_clips_rarv_smm_train \
        --v1-count 1667 --v2-count 1667 --v3-count 1666
"""

import argparse
import csv
import random
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def load_metadata(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def sample(metadata, n, seed=42):
    random.seed(seed)
    return metadata if len(metadata) <= n else random.sample(metadata, n)


def normalize_entry(entry, version, split):
    normalized = {
        "version": version, "split": split,
        "clip_filename": entry.get("clip_filename", ""),
        "duration": entry.get("duration", "0"),
        "target_duration": entry.get("target_duration", "0"),
        "audio_session": entry.get("audio_session", ""),
        "video_session": entry.get("video_session", ""),
        "source_audio_file": entry.get("source_audio_file", ""),
        "source_video_file": entry.get("source_video_file", ""),
        "class": entry.get("class", "RARV-SMM"),
        "processing_timestamp": entry.get("processing_timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    }
    if version == "V1":
        normalized.update(speaker_id=entry.get("speaker_id", ""),
                           audio_speaker_id=entry.get("speaker_id", ""),
                           video_speaker_id=entry.get("speaker_id", ""),
                           gender=entry.get("gender", ""), pairing_type="")
    elif version == "V2":
        normalized.update(speaker_id="", audio_speaker_id=entry.get("audio_speaker_id", ""),
                           video_speaker_id=entry.get("video_speaker_id", ""),
                           gender=entry.get("gender", ""), pairing_type="")
    else:  # V3
        normalized.update(speaker_id="", audio_speaker_id=entry.get("audio_speaker_id", ""),
                           video_speaker_id=entry.get("video_speaker_id", ""),
                           gender="", pairing_type=entry.get("pairing_type", ""))
    return normalized


def process_variant(metadata_entries, source_dir, version, output_dir, clip_id_counter, split):
    combined = []
    for entry in metadata_entries:
        source_filename = entry["clip_filename"]
        new_filename = f"rarv_smm_{version.lower()}_{clip_id_counter:05d}.mp4"
        source_path, dest_path = Path(source_dir) / source_filename, Path(output_dir) / new_filename
        if not source_path.exists():
            print(f"  Warning: source not found: {source_path}")
            continue
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest_path)

        normalized = normalize_entry(entry, version, split)
        normalized["clip_id"] = clip_id_counter
        normalized["clip_filename"] = new_filename
        combined.append(normalized)
        clip_id_counter += 1
    return combined, clip_id_counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--v1-dir", required=True)
    parser.add_argument("--v1-meta", required=True)
    parser.add_argument("--v2-dir", required=True)
    parser.add_argument("--v2-meta", required=True)
    parser.add_argument("--v3-dir", required=True)
    parser.add_argument("--v3-meta", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-meta", required=True)
    parser.add_argument("--v1-count", type=int, default=1667)
    parser.add_argument("--v2-count", type=int, default=1667)
    parser.add_argument("--v3-count", type=int, default=1666)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    v1 = sample(load_metadata(args.v1_meta), args.v1_count, seed=args.seed)
    v2 = sample(load_metadata(args.v2_meta), args.v2_count, seed=args.seed)
    v3 = sample(load_metadata(args.v3_meta), args.v3_count, seed=args.seed)

    counter = 1
    combined = []
    for entries, source_dir, version in [(v1, args.v1_dir, "V1"), (v2, args.v2_dir, "V2"), (v3, args.v3_dir, "V3")]:
        rows, counter = process_variant(entries, source_dir, version, args.output_dir, counter, args.split)
        combined.extend(rows)

    fieldnames = ["clip_id", "clip_filename", "version", "split", "audio_speaker_id", "video_speaker_id",
                  "speaker_id", "gender", "pairing_type", "audio_session", "video_session",
                  "source_audio_file", "source_video_file", "duration", "target_duration",
                  "class", "processing_timestamp"]
    with open(args.output_meta, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined)

    counts = defaultdict(int)
    for row in combined:
        counts[row["version"]] += 1
    print(f"Total clips: {len(combined)} (V1: {counts['V1']}, V2: {counts['V2']}, V3: {counts['V3']})")
    print(f"Videos -> {args.output_dir}, metadata -> {args.output_meta}")


if __name__ == "__main__":
    main()

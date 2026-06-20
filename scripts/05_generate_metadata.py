#!/usr/bin/env python3
"""
Phase 5 — Generate final metadata CSV for the processed RARV-SMM clips.

Verifies each clip from the pairing plan exists in output-dir, measures its
actual duration, and records source files, speakers/gender, and class label
('RARV-SMM') for downstream dataset construction (06_combine_rarv_smm_datasets.py).

Usage:
    python scripts/05_generate_metadata.py --variant v1 --plan pairing_plan.csv \
        --output-dir output_clips --output output_clips/rarv_smm_v1_metadata.csv \
        --speaker-metadata vox2_meta.csv
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.transforms import get_duration


def load_pairing_plan(plan_file):
    with open(plan_file, "r") as f:
        return list(csv.DictReader(f))


def load_speaker_gender(meta_file):
    gender = {}
    if not meta_file:
        return gender
    with open(meta_file, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            cleaned = {k.strip().strip("﻿"): v.strip() for k, v in row.items()}
            speaker_id = cleaned.get("VoxCeleb2 ID", "")
            if speaker_id:
                gender[speaker_id] = cleaned.get("Gender", "unknown")
    return gender


def generate_metadata(plan, output_dir: Path, variant: str, speaker_gender: dict):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for clip_data in plan:
        clip_id = int(clip_data["clip_id"])
        clip_filename = f"rarv_smm_{variant}_{clip_id:05d}.mp4"
        clip_path = output_dir / clip_filename
        if not clip_path.exists():
            continue

        duration = get_duration(clip_path) or float(clip_data.get("target_duration", 0))

        base = {
            "clip_id": clip_id, "clip_filename": clip_filename,
            "audio_session": clip_data["audio_session"], "video_session": clip_data["video_session"],
            "source_audio_file": clip_data["audio_file"], "source_video_file": clip_data["video_file"],
            "duration": duration, "target_duration": float(clip_data["target_duration"]),
            "class": "RARV-SMM", "version": variant.upper(), "processing_timestamp": timestamp,
        }
        if variant == "v1":
            base["speaker_id"] = clip_data["speaker_id"]
            base["gender"] = speaker_gender.get(clip_data["speaker_id"], "unknown")
        elif variant == "v2":
            base["audio_speaker_id"] = clip_data["audio_speaker"]
            base["video_speaker_id"] = clip_data["video_speaker"]
            base["gender"] = clip_data.get("gender", "unknown")
        else:
            base["audio_speaker_id"] = clip_data["audio_speaker"]
            base["video_speaker_id"] = clip_data["video_speaker"]
            base["pairing_type"] = clip_data.get("pairing_type", "unknown")
        rows.append(base)
    return rows


def save_metadata(rows, output_file, variant):
    if variant == "v1":
        fieldnames = ["clip_id", "clip_filename", "speaker_id", "gender", "audio_session", "video_session",
                      "source_audio_file", "source_video_file", "duration", "target_duration",
                      "class", "version", "processing_timestamp"]
    elif variant == "v2":
        fieldnames = ["clip_id", "clip_filename", "audio_speaker_id", "video_speaker_id", "gender",
                      "audio_session", "video_session", "source_audio_file", "source_video_file",
                      "duration", "target_duration", "class", "version", "processing_timestamp"]
    else:
        fieldnames = ["clip_id", "clip_filename", "audio_speaker_id", "video_speaker_id", "pairing_type",
                      "audio_session", "video_session", "source_audio_file", "source_video_file",
                      "duration", "target_duration", "class", "version", "processing_timestamp"]
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["v1", "v2", "v3"], required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--speaker-metadata", help="VoxCeleb2 speaker metadata CSV (for V1 gender lookup)")
    args = parser.parse_args()

    plan = load_pairing_plan(args.plan)
    speaker_gender = load_speaker_gender(args.speaker_metadata)
    rows = generate_metadata(plan, Path(args.output_dir), args.variant, speaker_gender)
    save_metadata(rows, args.output, args.variant)
    print(f"Generated metadata for {len(rows)} clips -> {args.output}")


if __name__ == "__main__":
    main()

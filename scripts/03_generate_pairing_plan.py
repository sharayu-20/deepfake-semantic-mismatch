#!/usr/bin/env python3
"""
Phase 3 — Generate pairing plan.

Turns the speaker/session selections from 02_filter_speakers.py into a CSV
pairing plan: one row per clip to be generated, specifying the source audio
file, source video file, and a random target duration in [3, 10] seconds.

Usage:
    python scripts/03_generate_pairing_plan.py --variant v1 \
        --selected selected_speakers.json --index dataset_index.json \
        --output pairing_plan.csv --num-clips 5000
"""

import argparse
import csv
import json
import random


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def build_index_lookup(index):
    return {(e["speaker_id"], e["session_id"]): e for e in index}


def generate_plan_v1(selected_speakers, index, num_clips, seed=42):
    random.seed(seed)
    lookup = build_index_lookup(index)

    all_pairs = []
    for speaker_data in selected_speakers:
        speaker_id = speaker_data["speaker_id"]
        for pair in speaker_data["pairs"]:
            audio_entry = lookup.get((speaker_id, pair["audio_session"]), {})
            video_entry = lookup.get((speaker_id, pair["video_session"]), {})
            if audio_entry.get("audio_files") and video_entry.get("video_files"):
                all_pairs.append({
                    "speaker_id": speaker_id,
                    "audio_session": pair["audio_session"],
                    "video_session": pair["video_session"],
                    "audio_files": audio_entry["audio_files"],
                    "video_files": video_entry["video_files"],
                })
    if not all_pairs:
        raise ValueError("No valid session pairs found")

    plan = []
    for clip_id in range(1, num_clips + 1):
        pair = all_pairs[(clip_id - 1) % len(all_pairs)]
        plan.append({
            "clip_id": clip_id,
            "audio_file": random.choice(pair["audio_files"])["path"],
            "video_file": random.choice(pair["video_files"])["path"],
            "target_duration": round(random.uniform(3.0, 10.0), 2),
            "speaker_id": pair["speaker_id"],
            "audio_session": pair["audio_session"],
            "video_session": pair["video_session"],
            "audio_duration": 0.0,
            "video_duration": 0.0,
        })
    return plan


def generate_plan_cross_identity(selected_pairs, index, num_clips, variant, seed=42):
    """Shared by V2 (gender field) and V3 (pairing_type field)."""
    random.seed(seed)
    lookup = build_index_lookup(index)

    all_pairs = []
    for pair_data in selected_pairs:
        audio_entry = lookup.get((pair_data["audio_speaker"], pair_data["audio_session"]), {})
        video_entry = lookup.get((pair_data["video_speaker"], pair_data["video_session"]), {})
        if audio_entry.get("audio_files") and video_entry.get("video_files"):
            entry = dict(pair_data, audio_files=audio_entry["audio_files"], video_files=video_entry["video_files"])
            all_pairs.append(entry)
    if not all_pairs:
        raise ValueError("No valid speaker pairs found")

    plan = []
    for clip_id in range(1, num_clips + 1):
        pair = all_pairs[(clip_id - 1) % len(all_pairs)]
        row = {
            "clip_id": clip_id,
            "audio_file": random.choice(pair["audio_files"])["path"],
            "video_file": random.choice(pair["video_files"])["path"],
            "target_duration": round(random.uniform(3.0, 10.0), 2),
            "audio_speaker": pair["audio_speaker"],
            "video_speaker": pair["video_speaker"],
            "audio_session": pair["audio_session"],
            "video_session": pair["video_session"],
            "audio_duration": 0.0,
            "video_duration": 0.0,
        }
        row["gender" if variant == "v2" else "pairing_type"] = pair.get("gender" if variant == "v2" else "pairing_type", "unknown")
        plan.append(row)
    return plan


def save_plan(plan, output_file, variant):
    if variant == "v1":
        fieldnames = ["clip_id", "audio_file", "video_file", "target_duration",
                      "speaker_id", "audio_session", "video_session", "audio_duration", "video_duration"]
    else:
        extra = "gender" if variant == "v2" else "pairing_type"
        fieldnames = ["clip_id", "audio_file", "video_file", "target_duration",
                      "audio_speaker", "video_speaker", "audio_session", "video_session",
                      extra, "audio_duration", "video_duration"]
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(plan)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["v1", "v2", "v3"], required=True)
    parser.add_argument("--selected", required=True, help="Output of 02_filter_speakers.py")
    parser.add_argument("--index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-clips", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    selected = load_json(args.selected)
    index = load_json(args.index)

    if args.variant == "v1":
        plan = generate_plan_v1(selected, index, args.num_clips, seed=args.seed)
    else:
        plan = generate_plan_cross_identity(selected, index, args.num_clips, args.variant, seed=args.seed)

    save_plan(plan, args.output, args.variant)
    print(f"Generated {len(plan)} clip pairings -> {args.output}")


if __name__ == "__main__":
    main()

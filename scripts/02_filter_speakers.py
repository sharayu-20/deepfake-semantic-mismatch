#!/usr/bin/env python3
"""
Phase 2 — Filter and select speaker/session pairs for RARV-SMM construction.

V1 (same identity, different context): keeps speakers with >= 4 sessions that have
both audio and video, then draws 2 disjoint (audio_session, video_session) pairs
per speaker from 4 distinct sessions.

V2 (same gender, different identity): pairs two different speakers of the same
gender, drawing audio from one and video from the other, excluding speakers
already used in V1 (--exclude-v1).

V3 (different gender, different identity): same as V2 but pairs speakers of
opposite genders (M->F / F->M), excluding speakers used in V1 and V2
(--exclude-v1, --exclude-v2).

Usage:
    python scripts/02_filter_speakers.py --variant v1 \
        --index dataset_index.json --output selected_speakers.json
    python scripts/02_filter_speakers.py --variant v2 \
        --index dataset_index.json --metadata vox2_meta.csv \
        --exclude-v1 output_clips/rarv_smm_v1_metadata.csv \
        --output selected_speakers_v2.json
    python scripts/02_filter_speakers.py --variant v3 \
        --index dataset_index.json --metadata vox2_meta.csv \
        --exclude-v1 output_clips/rarv_smm_v1_metadata.csv \
        --exclude-v2 output_clips_v2/rarv_smm_v2_metadata.csv \
        --output selected_speakers_v3.json
"""

import argparse
import csv
import json
import random
from collections import defaultdict


def load_index(index_file):
    with open(index_file, "r") as f:
        return json.load(f)


def load_metadata(meta_file):
    metadata = {}
    with open(meta_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned_row = {k.strip().strip("﻿"): v.strip() for k, v in row.items()}
            speaker_id = cleaned_row.get("VoxCeleb2 ID", "").strip()
            if speaker_id:
                metadata[speaker_id] = cleaned_row
    return metadata


def load_excluded_speakers(metadata_csv, speaker_id_keys):
    excluded = set()
    if not metadata_csv:
        return excluded
    try:
        with open(metadata_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key in speaker_id_keys:
                    value = row.get(key, "").strip()
                    if value:
                        excluded.add(value)
    except FileNotFoundError:
        pass
    return excluded


def filter_speakers_with_multiple_sessions(index, min_sessions=4):
    """V1: speakers with >= min_sessions sessions having both audio and video."""
    speaker_sessions = defaultdict(lambda: {"audio_sessions": set(), "video_sessions": set()})
    for entry in index:
        speaker_id = entry["speaker_id"]
        if entry["audio_files"]:
            speaker_sessions[speaker_id]["audio_sessions"].add(entry["session_id"])
        if entry["video_files"]:
            speaker_sessions[speaker_id]["video_sessions"].add(entry["session_id"])

    valid_speakers = []
    for speaker_id, sessions in speaker_sessions.items():
        if len(sessions["audio_sessions"]) >= min_sessions and len(sessions["video_sessions"]) >= min_sessions:
            common = sessions["audio_sessions"] & sessions["video_sessions"]
            if len(common) >= min_sessions:
                valid_speakers.append(speaker_id)
    return valid_speakers, speaker_sessions


def filter_speakers_with_any_session(index):
    """V2/V3: speakers with >= 1 session having both audio and video."""
    speaker_sessions = defaultdict(lambda: {"audio_sessions": set(), "video_sessions": set()})
    for entry in index:
        speaker_id = entry["speaker_id"]
        if entry["audio_files"]:
            speaker_sessions[speaker_id]["audio_sessions"].add(entry["session_id"])
        if entry["video_files"]:
            speaker_sessions[speaker_id]["video_sessions"].add(entry["session_id"])

    valid_speakers = [
        speaker_id for speaker_id, sessions in speaker_sessions.items()
        if len(sessions["audio_sessions"] & sessions["video_sessions"]) >= 1
    ]
    return valid_speakers, speaker_sessions


def select_speakers_and_pairs_v1(valid_speakers, speaker_sessions, index, num_speakers=2500, seed=42):
    random.seed(seed)
    selected_speakers = (random.sample(valid_speakers, num_speakers)
                          if len(valid_speakers) >= num_speakers else valid_speakers)

    index_lookup = defaultdict(lambda: defaultdict(dict))
    for entry in index:
        index_lookup[entry["speaker_id"]][entry["session_id"]] = entry

    selected_data = []
    for speaker_id in selected_speakers:
        common_sessions = list(speaker_sessions[speaker_id]["audio_sessions"] &
                                speaker_sessions[speaker_id]["video_sessions"])
        if len(common_sessions) < 4:
            continue

        s = random.sample(common_sessions, 4)
        pairs = [
            {"audio_session": s[0], "video_session": s[1]},
            {"audio_session": s[2], "video_session": s[3]},
        ]
        valid_pairs = [
            p for p in pairs
            if index_lookup[speaker_id].get(p["audio_session"], {}).get("audio_files")
            and index_lookup[speaker_id].get(p["video_session"], {}).get("video_files")
        ]
        if len(valid_pairs) == 2:
            selected_data.append({"speaker_id": speaker_id, "pairs": valid_pairs})
    return selected_data


def select_speakers_and_pairs_v2(valid_speakers, speaker_sessions, index, metadata, exclude,
                                  num_clips=5000, seed=42):
    """V2: same gender, different identity."""
    random.seed(seed)
    available = [s for s in valid_speakers if s not in exclude] or valid_speakers

    speakers_by_gender = {"m": [], "f": []}
    for speaker_id in available:
        gender = metadata.get(speaker_id, {}).get("Gender", "").strip().lower()
        if gender in ("m", "f"):
            speakers_by_gender[gender].append(speaker_id)

    index_lookup = defaultdict(lambda: defaultdict(dict))
    for entry in index:
        index_lookup[entry["speaker_id"]][entry["session_id"]] = entry

    total = len(speakers_by_gender["m"]) + len(speakers_by_gender["f"])
    weights = {g: len(speakers_by_gender[g]) / total for g in "mf"} if total else {"m": 0.5, "f": 0.5}

    selected_data = []
    for _ in range(num_clips // 2):
        if random.random() < weights["m"] and len(speakers_by_gender["m"]) >= 2:
            gender = "m"
        elif len(speakers_by_gender["f"]) >= 2:
            gender = "f"
        elif len(speakers_by_gender["m"]) >= 2:
            gender = "m"
        else:
            break

        audio_speaker, video_speaker = random.sample(speakers_by_gender[gender], 2)
        audio_common = list(speaker_sessions[audio_speaker]["audio_sessions"] & speaker_sessions[audio_speaker]["video_sessions"])
        video_common = list(speaker_sessions[video_speaker]["audio_sessions"] & speaker_sessions[video_speaker]["video_sessions"])
        if not audio_common or not video_common:
            continue

        audio_session, video_session = random.choice(audio_common), random.choice(video_common)
        if (index_lookup[audio_speaker].get(audio_session, {}).get("audio_files")
                and index_lookup[video_speaker].get(video_session, {}).get("video_files")):
            selected_data.append({
                "audio_speaker": audio_speaker, "video_speaker": video_speaker,
                "audio_session": audio_session, "video_session": video_session, "gender": gender,
            })
    return selected_data


def select_speakers_and_pairs_v3(valid_speakers, speaker_sessions, index, metadata, exclude,
                                  num_clips=5000, seed=42):
    """V3: different gender, different identity."""
    random.seed(seed)
    available = [s for s in valid_speakers if s not in exclude] or valid_speakers

    speakers_by_gender = {"m": [], "f": []}
    for speaker_id in available:
        gender = metadata.get(speaker_id, {}).get("Gender", "").strip().lower()
        if gender in ("m", "f"):
            speakers_by_gender[gender].append(speaker_id)
    if len(speakers_by_gender["m"]) < 1 or len(speakers_by_gender["f"]) < 1:
        raise ValueError("Not enough speakers of both genders for V3 (need >= 1 male and >= 1 female)")

    index_lookup = defaultdict(lambda: defaultdict(dict))
    for entry in index:
        index_lookup[entry["speaker_id"]][entry["session_id"]] = entry

    total = len(speakers_by_gender["m"]) + len(speakers_by_gender["f"])
    male_ratio = len(speakers_by_gender["m"]) / total if total else 0.5

    selected_data = []
    for _ in range(num_clips // 2):
        if random.random() < male_ratio:
            audio_gender, video_gender, pairing_type = "m", "f", "M->F"
        else:
            audio_gender, video_gender, pairing_type = "f", "m", "F->M"

        audio_speaker = random.choice(speakers_by_gender[audio_gender])
        video_speaker = random.choice(speakers_by_gender[video_gender])

        audio_common = list(speaker_sessions[audio_speaker]["audio_sessions"] & speaker_sessions[audio_speaker]["video_sessions"])
        video_common = list(speaker_sessions[video_speaker]["audio_sessions"] & speaker_sessions[video_speaker]["video_sessions"])
        if not audio_common or not video_common:
            continue

        audio_session, video_session = random.choice(audio_common), random.choice(video_common)
        if (index_lookup[audio_speaker].get(audio_session, {}).get("audio_files")
                and index_lookup[video_speaker].get(video_session, {}).get("video_files")):
            selected_data.append({
                "audio_speaker": audio_speaker, "video_speaker": video_speaker,
                "audio_session": audio_session, "video_session": video_session,
                "pairing_type": pairing_type,
            })
    return selected_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["v1", "v2", "v3"], required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--metadata", help="VoxCeleb2 speaker metadata CSV (required for v2/v3)")
    parser.add_argument("--exclude-v1", help="output_clips*/rarv_smm_v1*_metadata.csv (v2/v3)")
    parser.add_argument("--exclude-v2", help="output_clips_v2*/rarv_smm_v2*_metadata.csv (v3 only)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-speakers", type=int, default=2500, help="v1 only")
    parser.add_argument("--num-clips", type=int, default=5000, help="v2/v3 only")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    index = load_index(args.index)

    if args.variant == "v1":
        valid_speakers, speaker_sessions = filter_speakers_with_multiple_sessions(index)
        selected = select_speakers_and_pairs_v1(valid_speakers, speaker_sessions, index,
                                                 num_speakers=args.num_speakers, seed=args.seed)
    else:
        metadata = load_metadata(args.metadata)
        valid_speakers, speaker_sessions = filter_speakers_with_any_session(index)
        exclude = load_excluded_speakers(args.exclude_v1, ["speaker_id", "audio_speaker_id", "video_speaker_id"])
        if args.variant == "v3":
            exclude |= load_excluded_speakers(args.exclude_v2, ["audio_speaker_id", "video_speaker_id"])
            selected = select_speakers_and_pairs_v3(valid_speakers, speaker_sessions, index, metadata,
                                                      exclude, num_clips=args.num_clips, seed=args.seed)
        else:
            selected = select_speakers_and_pairs_v2(valid_speakers, speaker_sessions, index, metadata,
                                                      exclude, num_clips=args.num_clips, seed=args.seed)

    with open(args.output, "w") as f:
        json.dump(selected, f, indent=2)
    print(f"Selected {len(selected)} entries for {args.variant} -> {args.output}")


if __name__ == "__main__":
    main()

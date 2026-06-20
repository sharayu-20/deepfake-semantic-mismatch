#!/usr/bin/env python3
"""
Phase 1 — Index VoxCeleb2.

Scans the VoxCeleb2 audio (.m4a) and video (.mp4) trees and builds a lightweight
JSON index of {speaker_id, session_id, audio_files, video_files}, without decoding
media files. This index is the input to 02_filter_speakers.py.

Usage:
    python scripts/01_index_voxceleb2.py \
        --audio-dir /path/to/vox2_dev_aac/dev/aac \
        --video-dir /path/to/vox2_dev_mp4/dev/mp4 \
        --output    dataset_index.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


def scan_directory(base_path, extension):
    files_by_speaker_session = defaultdict(lambda: defaultdict(list))
    base_path = Path(base_path)
    if not base_path.exists():
        print(f"Warning: directory {base_path} does not exist")
        return files_by_speaker_session

    files = list(base_path.glob(f"**/*.{extension}"))
    print(f"Found {len(files)} {extension} files")

    base_parts_len = len(base_path.parts)
    for file_path in tqdm(files, desc=f"Scanning {extension} files"):
        parts = file_path.parts
        if base_parts_len < len(parts) - 1:
            speaker_id, session_id = parts[base_parts_len], parts[base_parts_len + 1]
            files_by_speaker_session[speaker_id][session_id].append(file_path)
    return files_by_speaker_session


def build_index(audio_base, video_base):
    audio_files = scan_directory(audio_base, "m4a")
    video_files = scan_directory(video_base, "mp4")

    index = []
    all_speakers = set(audio_files.keys()) | set(video_files.keys())
    for speaker_id in tqdm(sorted(all_speakers), desc="Indexing speakers"):
        audio_sessions = audio_files.get(speaker_id, {})
        video_sessions = video_files.get(speaker_id, {})
        for session_id in set(audio_sessions.keys()) | set(video_sessions.keys()):
            audio_list = [{"path": str(p)} for p in audio_sessions.get(session_id, [])]
            video_list = [{"path": str(p)} for p in video_sessions.get(session_id, [])]
            if audio_list or video_list:
                index.append({
                    "speaker_id": speaker_id,
                    "session_id": session_id,
                    "audio_files": audio_list,
                    "video_files": video_list,
                })
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    index = build_index(args.audio_dir, args.video_dir)
    with open(args.output, "w") as f:
        json.dump(index, f, indent=2)

    speakers = {entry["speaker_id"] for entry in index}
    print(f"Unique speakers: {len(speakers)} | sessions: {len(index)} | saved to {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Phase 4 — Process clips: standardize video/audio and mux into final RARV-SMM clips.

For each row of the pairing plan (03_generate_pairing_plan.py):
  - video: resize to 224x224, 25 fps, extract a target_duration segment at a random start
  - audio: mono, 16kHz, loudness-normalized to -23 LUFS, truncated/looped to target_duration
  - mux into rarv_smm_{variant}_{clip_id:05d}.mp4 (video stream + mismatched audio stream)

Uses utils.transforms for the underlying ffmpeg calls.

Usage:
    python scripts/04_process_clips.py --variant v1 --plan pairing_plan.csv \
        --output-dir output_clips
"""

import argparse
import csv
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.transforms import combine_audio_video, get_duration, process_audio, process_video


def load_pairing_plan(plan_file):
    with open(plan_file, "r") as f:
        return list(csv.DictReader(f))


def process_clip(clip_data, output_dir: Path, variant: str, seed: int):
    clip_id = int(clip_data["clip_id"])
    # Local RNG keyed by clip_id: the global `random` module isn't ordering-safe
    # across ThreadPoolExecutor workers, so a shared `random.seed()` call in main()
    # would make start_time depend on nondeterministic thread scheduling.
    rng = random.Random(seed + clip_id)
    final_clip = output_dir / f"rarv_smm_{variant}_{clip_id:05d}.mp4"
    if final_clip.exists():
        return clip_id, True, "Already exists"

    audio_file, video_file = clip_data["audio_file"], clip_data["video_file"]
    target_duration = float(clip_data["target_duration"])

    audio_duration = float(clip_data.get("audio_duration") or 0) or get_duration(audio_file)
    video_duration = float(clip_data.get("video_duration") or 0) or get_duration(video_file)
    if not audio_duration or not video_duration:
        return clip_id, False, "Could not determine source durations"

    temp_video = output_dir / f"video_{clip_id:05d}.mp4"
    temp_audio = output_dir / f"audio_{clip_id:05d}.wav"

    max_start = max(0.0, video_duration - target_duration)
    start_time = rng.uniform(0, max_start) if max_start > 0 else 0.0

    if not process_video(video_file, temp_video, target_duration, video_duration, start_time):
        return clip_id, False, "Video processing failed"
    if not process_audio(audio_file, temp_audio, target_duration, audio_duration):
        temp_video.unlink(missing_ok=True)
        return clip_id, False, "Audio processing failed"

    success = combine_audio_video(temp_video, temp_audio, final_clip)
    temp_video.unlink(missing_ok=True)
    temp_audio.unlink(missing_ok=True)
    if not success:
        final_clip.unlink(missing_ok=True)
        return clip_id, False, "Mux failed"
    return clip_id, True, "Success"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["v1", "v2", "v3"], required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=min(cpu_count() * 2, 16))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = load_pairing_plan(args.plan)

    success_count, fail_count = 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_clip, row, output_dir, args.variant, args.seed): row for row in plan}
        for i, future in enumerate(as_completed(futures), start=1):
            clip_id, success, message = future.result()
            success_count += success
            fail_count += not success
            if not success:
                print(f"  Clip {clip_id} failed: {message}")
            if i % 50 == 0:
                print(f"Processed {i}/{len(plan)} (success: {success_count}, failed: {fail_count})")

    print(f"Done. Success: {success_count}, Failed: {fail_count}. Output: {output_dir}")


if __name__ == "__main__":
    main()

"""
ffmpeg-based preprocessing used to build RARV-SMM clips from VoxCeleb2:
  - video: resize to 224x224, 25 fps, extract a 3-10s segment
  - audio: mono, 16 kHz, loudness-normalised to -23 LUFS, duration-aligned to video
           (truncated if longer, looped + atempo-adjusted if shorter)
  - mux the standardised video with the standardised (semantically mismatched) audio

Used by scripts/04_process_clips.py.
"""

import subprocess
from pathlib import Path


class FFmpegNotFoundError(RuntimeError):
    pass


def find_ffmpeg(tool: str = "ffmpeg"):
    for path in [tool, f"/usr/bin/{tool}", f"/usr/local/bin/{tool}"]:
        result = subprocess.run(["which", path], capture_output=True)
        if result.returncode == 0:
            return path
    return None


def require_ffmpeg(tool: str = "ffmpeg") -> str:
    path = find_ffmpeg(tool)
    if path is None:
        raise FFmpegNotFoundError(
            f"'{tool}' not found on PATH or in /usr/bin, /usr/local/bin. "
            f"Install ffmpeg (e.g. `brew install ffmpeg` or `apt install ffmpeg`) before running this pipeline."
        )
    return path


def get_duration(file_path) -> float | None:
    ffprobe = require_ffmpeg("ffprobe")
    cmd = [ffprobe, "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", str(file_path)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            duration = float(result.stdout.strip())
            return duration if duration > 0 else None
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
        pass
    return None


def process_video(input_path, output_path, target_duration: float, video_duration: float, start_time: float) -> bool:
    """Resize to 224x224 @ 25fps and extract a target_duration segment starting at start_time."""
    ffmpeg = require_ffmpeg()
    start_time_str = f"{start_time:.6f}".rstrip("0").rstrip(".") or "0"
    cmd = [
        ffmpeg, "-y", "-i", str(input_path),
        "-ss", start_time_str,
        "-t", str(target_duration),
        "-vf", "scale=224:224",
        "-r", "25",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def process_audio(input_path, output_path, target_duration: float, audio_duration: float) -> bool:
    """Mono, 16kHz, loudness-normalised to -23 LUFS; truncated or looped to target_duration."""
    ffmpeg = require_ffmpeg()
    if audio_duration >= target_duration:
        cmd = [
            ffmpeg, "-y", "-i", str(input_path),
            "-ac", "1", "-ar", "16000",
            "-af", "loudnorm=I=-23:TP=-2.0:LRA=7.0",
            "-t", str(target_duration),
            str(output_path),
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True)
            return True
        except subprocess.CalledProcessError:
            return False

    # Audio shorter than target: normalise, then loop with atempo-safe stream_loop
    output_path = Path(output_path)
    temp_wav = output_path.parent / f"temp_{output_path.name}"
    norm_cmd = [
        ffmpeg, "-y", "-i", str(input_path),
        "-ac", "1", "-ar", "16000",
        "-af", "loudnorm=I=-23:TP=-2.0:LRA=7.0",
        str(temp_wav),
    ]
    try:
        subprocess.run(norm_cmd, capture_output=True, text=True, timeout=60, check=True)
    except subprocess.CalledProcessError:
        return False

    if audio_duration <= 0:
        temp_wav.unlink(missing_ok=True)
        return False

    loops_needed = int(target_duration / audio_duration) + 1
    loop_cmd = [
        ffmpeg, "-y",
        "-stream_loop", str(loops_needed),
        "-i", str(temp_wav),
        "-t", str(target_duration),
        "-ac", "1", "-ar", "16000",
        str(output_path),
    ]
    try:
        subprocess.run(loop_cmd, capture_output=True, text=True, timeout=60, check=True)
        success = True
    except subprocess.CalledProcessError:
        success = False
    temp_wav.unlink(missing_ok=True)
    return success


def combine_audio_video(video_path, audio_path, output_path) -> bool:
    """Mux the standardised video stream with the standardised (mismatched) audio stream."""
    ffmpeg = require_ffmpeg()
    cmd = [
        ffmpeg, "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-map", "0:v",
        "-map", "1:a",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "128k",
        "-shortest",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

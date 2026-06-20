#!/usr/bin/env python3
"""
Run five-class semantic-reinforced inference on a single video.

Extracts frames + audio, computes the frozen ImageBind semantic coherence
score, and runs the chosen backbone to predict one of
{RARV, RAFV, FARV, FAFV, RARV-SMM}.

Usage:
    python scripts/inference.py --model fgmdf --checkpoint summary/weight/5class_fgmdf_v1_imagebind/12.pth \
        --frame_dir /path/to/extracted_frames --audio_path /path/to/audio.wav
"""

import argparse
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.semantic_scorer import ImageBindSemanticScorer

CLASS_NAMES = ["RARV", "RAFV", "FARV", "FAFV", "RARV-SMM"]


def build_model(model_name, num_classes=5):
    if model_name == "fgmdf":
        from models.fgmdf_semantic import GAT_video_audio_semantic_v3
        return GAT_video_audio_semantic_v3(num_classes=num_classes, audio_nodes=4)
    if model_name == "fgi":
        from models.fgi_semantic import My_Network_Semantic
        return My_Network_Semantic(num_classes=num_classes)
    raise ValueError(f"Unsupported model: {model_name}")


def load_clip_tensors(frame_dir, image_size=128, num_frame=4):
    """
    Loads frames/audio using the backbone's own Multimodal_dataset preprocessing
    (dataset.dataset.Multimodal_dataset), so normalization matches training exactly.

    Multimodal_dataset is built from a txt file of "<frame_path> <label> <start_second>"
    lines; for single-clip inference we write one such line to a temp file and read
    item 0 back out.
    """
    import tempfile

    from dataset.dataset import Multimodal_dataset

    frame_files = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
    first_frame = frame_files[0]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
        tmp.write(f"{first_frame} 0 0\n")
        tmp_path = tmp.name

    try:
        loader = Multimodal_dataset(image_size, "test", tmp_path, num_frame=num_frame)
        _, img_data, aud_data, _, _, _ = loader[0]
    finally:
        os.unlink(tmp_path)

    return img_data.unsqueeze(0), aud_data.unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["fgmdf", "fgi"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frame_dir", required=True, help="Directory of extracted frame PNGs")
    parser.add_argument("--audio_path", required=True)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--num_frame", type=int, default=4)
    parser.add_argument("--gpu", type=str, default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    scorer = ImageBindSemanticScorer(device=device)
    frame_files = sorted(glob.glob(os.path.join(args.frame_dir, "*.png")))
    semantic_score = torch.tensor([[scorer.score(args.audio_path, frame_files)]], dtype=torch.float32).to(device)

    video, audio = load_clip_tensors(args.frame_dir, args.image_size, args.num_frame)
    video, audio = video.to(device), audio.to(device)

    model = build_model(args.model).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        logits, *_ = model(video, audio, semantic_score)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu()

    pred_class = CLASS_NAMES[int(torch.argmax(probs))]
    print(f"Semantic coherence score: {semantic_score.item():.4f}")
    print(f"Predicted class: {pred_class}")
    for name, p in zip(CLASS_NAMES, probs.tolist()):
        print(f"  {name:<12} {p:.4f}")


if __name__ == "__main__":
    main()

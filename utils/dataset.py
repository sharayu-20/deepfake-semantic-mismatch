"""
Multimodal_dataset_semantic / collate_fn_semantic — dataset utilities for the
five-class (RARV, RAFV, FARV, FAFV, RARV-SMM) setting with ImageBind semantic
scores attached.

Multimodal_dataset_semantic wraps the FGMDF backbone's Multimodal_dataset
(dataset.dataset.Multimodal_dataset) and injects a pre-computed cosine
similarity score (produced by scripts/compute_semantic_scores.py) keyed by the
frame directory path. Returns a 7-tuple:

    (filename, img_data, aud_data, semantic_score, total_label, video_label, audio_label)
"""

import json

import torch
import torch.nn.functional as F

from dataset.dataset import Multimodal_dataset


class Multimodal_dataset_semantic(Multimodal_dataset):
    def __init__(self, image_size, type, txt_path, num_frame, scores_json):
        super().__init__(image_size, type, txt_path, num_frame)

        with open(scores_json, "r") as f:
            self.semantic_scores = json.load(f)

    def __getitem__(self, index):
        filename, img_data, aud_data, total_label, video_label, audio_label = \
            super().__getitem__(index)

        score = self.semantic_scores.get(filename, 0.5)
        semantic_score = torch.tensor([score], dtype=torch.float32)  # [1]

        return filename, img_data, aud_data, semantic_score, total_label, video_label, audio_label


def collate_fn_semantic(batch, target_audio_len: int = 64000):
    """
    Collate function for the 7-tuple produced by Multimodal_dataset_semantic.

    Returns:
        names                list[str]
        video_batch           [B, 120, H, W]  float32
        audio_batch           [B, 64000]      float32 (padded)
        semantic_score_batch  [B, 1]          float32
        total_label_batch     [B]             long
        video_label_batch     [B]             long
        audio_label_batch     [B]             long
    """
    names = [item[0] for item in batch]
    videos = [item[1] for item in batch]
    audios = [item[2] for item in batch]
    sem_scores = [item[3] for item in batch]
    total_labels = [item[4] for item in batch]
    video_labels = [item[5] for item in batch]
    audio_labels = [item[6] for item in batch]

    video_batch = torch.stack(videos, dim=0)
    semantic_score_batch = torch.stack(sem_scores, dim=0)

    pad_length = max(max(audio.shape[0] for audio in audios), target_audio_len)
    padded_audios = []
    for audio in audios:
        if audio.shape[0] < pad_length:
            padding = pad_length - audio.shape[0]
            pad_dims = (0, padding) if audio.dim() == 1 else (0, 0, 0, padding)
            padded_audios.append(F.pad(audio, pad_dims, mode="constant", value=0))
        else:
            padded_audios.append(audio[:pad_length])
    audio_batch = torch.stack(padded_audios, dim=0)

    return (
        names,
        video_batch,
        audio_batch,
        semantic_score_batch,
        torch.tensor(total_labels, dtype=torch.long),
        torch.tensor(video_labels, dtype=torch.long),
        torch.tensor(audio_labels, dtype=torch.long),
    )

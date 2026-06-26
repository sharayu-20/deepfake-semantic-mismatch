"""
GAT_video_audio_semantic_v3 — extends the FGMDF graph-attention backbone [Yin et al.,
IJCV 2024] with the ImageBind semantic coherence score.

Only change from the base model:
  - mix_pre (256 -> num_classes) replaced by mix_pre_semantic (257 -> num_classes)
  - forward() takes an extra semantic_score [B, 1] argument
  - The score is concatenated to fusion_out before final classification

Loss is unchanged: loss1 (5-class CE) + loss2 (binary video CE) + loss3 (binary audio CE).
No new loss term is needed; the semantic signal is injected directly into the features.

Requires the FGMDF backbone (network.graph_video_audio_model.GAT_video_audio) from
https://github.com/yinql1995/Fine-grained-Multimodal-DeepFake-Classification on PYTHONPATH.
"""

import torch
import torch.nn as nn

from network.graph_video_audio_model import GAT_video_audio


class GAT_video_audio_semantic_v3(GAT_video_audio):
    def __init__(self, num_classes: int = 5, audio_nodes: int = 4):
        super().__init__(num_classes=num_classes, audio_nodes=audio_nodes)
        self.mix_pre_semantic = nn.Linear(256 + 1, num_classes)

    def forward(self, vid_inp, aud_inp, semantic_score):
        """
        Args:
            vid_inp:        video tensor  [B, 120, H, W]
            aud_inp:        audio tensor  [B, 64000]
            semantic_score: ImageBind cosine similarity [B, 1] (pre-computed, frozen)

        Returns:
            mix_out_semantic: [B, num_classes]  5-class logits (semantic-enriched)
            video_out:        [B, 2]            binary video authenticity
            audio_out:        [B, 2]            binary audio authenticity
            fusion_out:       [B, 256]           raw fusion features (for visualization)
        """
        _, video_out, audio_out, fusion_out = super().forward(vid_inp, aud_inp)

        fusion_with_sem = torch.cat([fusion_out, semantic_score], dim=1)  # [B, 257]
        mix_out_semantic = self.mix_pre_semantic(fusion_with_sem)

        return mix_out_semantic, video_out, audio_out, fusion_out

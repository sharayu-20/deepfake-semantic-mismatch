"""
AVDF_Multiclass_Semantic — extends the AVDF/AV-HuBERT backbone [Khalid et al., 2021;
Shi et al., 2022] with the ImageBind semantic coherence score.

Only change from AVDF_Multiclass: a pre-computed ImageBind cosine similarity score
(one scalar per sample, [B, 1]) is concatenated to the 768-d AV-HuBERT fusion
embedding before the final classifier:

    Linear(769 -> 768) -> ReLU -> Linear(768 -> 5)

Everything else -- AV-HuBERT feature extractor, transformer fusion encoder, CE loss,
Lightning training/eval steps, k-fold logic -- is identical to AVDF_Multiclass.

Requires the AVDF/MRDF backbone (model.avdf_multiclass.AVDF_Multiclass) from
https://github.com/Vincent-ZHQ/MRDF on PYTHONPATH.
"""

import torch
import torch.nn as nn
from torch import Tensor

from model.avdf_multiclass import AVDF_Multiclass


class AVDF_Multiclass_Semantic(AVDF_Multiclass):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mm_classifier_semantic = nn.Sequential(
            nn.Linear(self.embed + 1, self.embed),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed, 5),
        )

    def forward(self, video: Tensor, audio: Tensor, mask: Tensor, semantic_score: Tensor) -> Tensor:
        if audio is not None:
            a_features = self.feature_extractor_audio_hubert(audio).transpose(1, 2)
        else:
            v_tmp = self.feature_extractor_video_hubert(video).transpose(1, 2)
            a_features = torch.zeros_like(v_tmp)

        v_features = self.feature_extractor_video_hubert(video).transpose(1, 2)
        av_features = torch.cat([a_features, v_features], dim=2)

        a_features = self.project_audio(a_features)
        v_features = self.project_video(v_features)
        av_features = self.project_hubert(av_features)

        av_features, _ = self.fusion_encoder_hubert(av_features, padding_mask=mask)
        fusion_vec = av_features[:, 0, :]  # [B, 768]

        sem = semantic_score.to(fusion_vec.device)  # [B, 1]
        fused = torch.cat([fusion_vec, sem], dim=1)  # [B, 769]
        return self.mm_classifier_semantic(fused)

    # Step overrides — only change is passing semantic_score through to forward()

    def training_step(self, batch, batch_idx=None, optimizer_idx=None, hiddens=None):
        return self._step(batch, log_prefix="train")

    def validation_step(self, batch, batch_idx=None, optimizer_idx=None, hiddens=None):
        return self._step(batch, log_prefix="val", loss_key="mm_loss")

    def test_step(self, batch, batch_idx=None, optimizer_idx=None, hiddens=None):
        return self._step(batch, log_prefix=None, loss_key="mm_loss")

    def _step(self, batch, log_prefix=None, loss_key="loss"):
        if not batch or "video" not in batch or batch["video"] is None or len(batch["video"]) == 0:
            dummy = sum(p.sum() * 0.0 for p in self.parameters() if p.requires_grad)
            return {"loss": dummy, "preds": None, "targets": None, "skip": True}

        m_logits = self(batch["video"], batch["audio"], batch["padding_mask"], batch["semantic_score"])
        loss_dict = self.loss_fn(m_logits, batch["mm_label"])

        preds = torch.argmax(self.softmax(m_logits), dim=1)
        probs = self.softmax(m_logits)

        if log_prefix:
            self.log_dict({f"{log_prefix}_{k}": v for k, v in loss_dict.items()},
                          on_step=True, on_epoch=True, prog_bar=False, sync_dist=self.distributed)

        return {"loss": loss_dict[loss_key], "preds": preds.detach(),
                "probs": probs.detach(), "targets": batch["mm_label"].detach()}

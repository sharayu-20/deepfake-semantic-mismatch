"""
FakeavcelebSemantic / FakeavcelebSemanticDataModule — AVDF-specific dataset
wrapper that attaches pre-computed ImageBind semantic scores to each sample.

Wraps the AVDF/MRDF backbone's dataset.fakeavceleb.{Fakeavceleb, FakeavcelebDataModule}.
JSON key format matches `meta.path`, e.g.
'FakeAVCeleb/RealVideo-RealAudio/African/men/id00166'. Samples missing from the
precomputed JSON (e.g. RARV-SMM samples not yet scored) default to 0.5.

Requires the AVDF/MRDF backbone (dataset.fakeavceleb) from
https://github.com/Zhixi-Cai/MRDF on PYTHONPATH.
"""

import json

import torch

from dataset.fakeavceleb import Fakeavceleb, FakeavcelebDataModule


class FakeavcelebSemantic(Fakeavceleb):
    def __init__(self, subset, root, metadata, scores: dict):
        super().__init__(subset, root, metadata)
        self.scores = scores

    def __getitem__(self, index: int):
        result = super().__getitem__(index)
        if result["id"] is None:
            result["semantic_score"] = torch.tensor([0.5], dtype=torch.float32)
            return result

        meta = self.metadata.iloc[index]
        score = float(self.scores.get(meta.path, 0.5))
        result["semantic_score"] = torch.tensor([score], dtype=torch.float32)
        return result


class FakeavcelebSemanticDataModule(FakeavcelebDataModule):
    def __init__(self, root, scores_json: str, **kwargs):
        super().__init__(root=root, **kwargs)
        with open(scores_json) as f:
            self.scores = json.load(f)

    def setup(self, stage=None):
        super().setup(stage)
        self.train_dataset = FakeavcelebSemantic("train", self.root, self.train_metadata, self.scores)
        self.test_dataset = FakeavcelebSemantic("test", self.root, self.test_metadata, self.scores)

    def collater(self, samples):
        batch = super().collater(samples)
        if not batch:
            return batch

        valid_ids = set(batch["id"].tolist())
        sem_scores = [s["semantic_score"] for s in samples if s["id"] is not None and s["id"] in valid_ids]
        if sem_scores:
            batch["semantic_score"] = torch.stack(sem_scores, dim=0)
        else:
            batch["semantic_score"] = torch.full((len(batch["id"]), 1), 0.5, dtype=torch.float32)
        return batch

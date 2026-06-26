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

        # IMPORTANT: the base MRDF collater (dataset.fakeavceleb.FakeavcelebDataModule.collater,
        # not vendored in this repo) may reorder or drop samples relative to `samples`
        # (e.g. sorting by sequence length for padding) -- so we cannot assume
        # batch["id"] is in the same order as `samples`. Build an id -> score lookup
        # first, then re-derive the score list in batch["id"]'s actual order, rather
        # than filtering `samples` positionally. Verify this against the real MRDF
        # collater() if you change backbone versions.
        score_by_id = {s["id"]: s["semantic_score"] for s in samples if s["id"] is not None}
        sem_scores = [score_by_id.get(int(sample_id), torch.tensor([0.5], dtype=torch.float32))
                      for sample_id in batch["id"].tolist()]
        assert len(sem_scores) == len(batch["id"]), (
            f"semantic_score count ({len(sem_scores)}) != batch size ({len(batch['id'])}) "
            "-- the base collater's id ordering assumption no longer holds."
        )
        batch["semantic_score"] = torch.stack(sem_scores, dim=0)
        return batch

"""
Frozen ImageBind audio-visual semantic coherence scorer.

For each sample, extracts a 1024-d audio embedding and a 1024-d video embedding
(mean of 4 evenly-spaced frames) from a frozen ImageBind model, and returns the
cosine similarity remapped from [-1, 1] to [0, 1]:

    s = 0.5 * (cos(e_a, e_v) + 1)

s = 0   -> maximum semantic divergence
s = 0.5 -> neutral
s = 1   -> full semantic alignment

ImageBind is never fine-tuned and is not involved in backpropagation; this score
is pre-computed offline and consumed as a single extra input feature by the
semantic-reinforced classifiers in fgmdf_semantic.py, fgi_semantic.py and
avdf_semantic.py.

Requires the official ImageBind package: https://github.com/facebookresearch/ImageBind
"""

import glob
import os

import torch
import torch.nn.functional as F


class ImageBindSemanticScorer:
    def __init__(self, device: str = "cuda"):
        from imagebind import data as ib_data
        from imagebind.models import imagebind_model

        self.device = device if torch.cuda.is_available() else "cpu"
        self.ib_data = ib_data
        self.model = imagebind_model.imagebind_huge(pretrained=True)
        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def score(self, audio_path: str, frame_paths: list) -> float:
        """Cosine similarity between one audio clip and its video frames, in [0, 1]."""
        if not frame_paths or not os.path.exists(audio_path):
            return 0.5

        audio_inputs = {"audio": self.ib_data.load_and_transform_audio_data([audio_path], self.device)}
        audio_emb = self.model(audio_inputs)["audio"]

        n = len(frame_paths)
        selected = [frame_paths[n * i // 4] for i in range(4)]
        frame_embs = []
        for fp in selected:
            vis_inputs = {"vision": self.ib_data.load_and_transform_vision_data([fp], self.device)}
            frame_embs.append(self.model(vis_inputs)["vision"])
        video_emb = torch.stack(frame_embs, dim=0).mean(dim=0)

        sim = (F.cosine_similarity(audio_emb, video_emb).item() + 1) / 2
        return sim

    @torch.no_grad()
    def score_directory(self, frame_dir: str, audio_path: str) -> float:
        frame_files = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
        return self.score(audio_path, frame_files)

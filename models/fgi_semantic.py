"""
My_Network_Semantic — extends the FGI distance-based backbone [Astrid et al.,
arXiv:2408.06753] with the ImageBind semantic coherence score.

Only change from My_Network: the final classifier concatenates the pre-computed
semantic score [B, 1] to the 784-d audio-visual spatial distance descriptor,
giving Linear(785, num_classes). All backbone, fusion, and attention code is
inherited unchanged.

Requires the FGI backbone (model.My_Network) from
https://github.com/aseuteurideu/FGI on PYTHONPATH.
"""

import torch
import torch.nn as nn

from model import My_Network


class My_Network_Semantic(My_Network):
    def __init__(self, network="resnet18", with_attention=False,
                 residual_conn=False, spatial_size=28, num_classes=5):
        super().__init__(network, with_attention, residual_conn, spatial_size, num_classes)

        map_size = spatial_size * spatial_size  # 784 for spatial_size=28

        if num_classes == 5:
            self.final_fc_semantic = nn.Linear(map_size + 1, num_classes)
        else:
            self.final_fc_semantic = nn.Sequential(
                nn.Linear(map_size + 1, 1),
                nn.Sigmoid(),
            )

    def forward(self, vid_seq, aud_seq, semantic_score):
        """semantic_score: [B, 1] pre-computed ImageBind cosine similarity in [0, 1]."""
        vid_out = self.forward_lip(vid_seq)   # [B, 128, 15, 28, 28]
        aud_out = self.forward_aud(aud_seq)   # [B, 128, 15]

        aud_out = aud_out.view(aud_out.shape[0], aud_out.shape[1], aud_out.shape[2], 1, 1)

        vid_aud_distance_ = torch.pow((vid_out - aud_out), 2)
        vid_aud_distance_ = vid_aud_distance_.view(
            vid_aud_distance_.shape[0],
            vid_aud_distance_.shape[1] * vid_aud_distance_.shape[2],
            vid_aud_distance_.shape[3] * vid_aud_distance_.shape[4],
        )
        vid_aud_distance_ = torch.sqrt(torch.sum(vid_aud_distance_, dim=1))  # [B, 784]

        if self.with_attention:
            img_emb = self.img_emb_layer(vid_out)
            aud_emb = self.aud_emb_layer(
                aud_out.view(aud_out.shape[0], aud_out.shape[1], aud_out.shape[2])
            )

            atts = []
            for i_emb, a_emb in zip(img_emb, aud_emb):
                atts.append(torch.tensordot(i_emb, a_emb, dims=([0, 1], [0, 1])))
            att = torch.stack(atts)
            att = att / (32 * 15)
            att = att.view(att.shape[0], -1)
            att = torch.nn.functional.softmax(att, dim=1)

            if self.residual_conn:
                vid_aud_distance_ = torch.mul(att, vid_aud_distance_) + vid_aud_distance_
            else:
                vid_aud_distance_ = torch.mul(att, vid_aud_distance_)
        else:
            att = None

        features = torch.cat([vid_aud_distance_, semantic_score], dim=1)  # [B, 785]
        final_out = self.final_fc_semantic(features)

        return final_out, vid_aud_distance_, att

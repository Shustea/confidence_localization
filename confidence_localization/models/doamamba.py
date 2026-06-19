"""DOAMAMBA: the RTF-front-end Mamba model (the original architecture).

Backbone + head are byte-for-byte identical to the pre-refactor ``train.py``
class, so existing checkpoints load unchanged. All loss/metric/training logic
lives in :class:`models.base.DOABase`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .base import DOABase
from .blocks import MambaResCTF
from .losses import gaussian_loss, von_mises_loss


class DOAMAMBA(DOABase):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.strict_loading = False

        self.negative_log_likelihood_func = [
            (
                gaussian_loss
                if cfg.nnl_func == "gaussian"
                else von_mises_loss if cfg.nnl_func == "von_mises" else None
            )
        ][0]

        self.mamba_layers = nn.Sequential(
            *[MambaResCTF(cfg, expand) for expand in cfg.layers]
        )

        head_in = int(cfg.freq_dim) * int(cfg.d_model)
        head_hidden = int(getattr(cfg, "head_hidden", 256))
        self.head = nn.Sequential(
            nn.Linear(head_in, head_hidden),
            nn.GELU(),
            nn.Dropout(float(getattr(cfg, "head_dropout", 0.1))),
        )

        self.doa = nn.Sequential(
            nn.Linear(head_hidden, 2), nn.Tanh()  # output in [-1, 1]
        )

        self.log_std = nn.Linear(head_hidden, 1)
        self.log_std_min = float(getattr(cfg, "log_std_min", -4.0))  # sigma ~ 1.0 deg
        self.log_std_max = float(getattr(cfg, "log_std_max", 1.5))  # sigma ~ 256 deg

        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.head[0].weight)
        init.zeros_(self.head[0].bias)

    def forward(self, x):
        x = x.permute(0, -1, 2, 1).contiguous()  # [B, freq, T, d]

        for block in self.mamba_layers:
            x = block(F.normalize(x, dim=-1))

        B, Fq, T, d = x.shape
        # [B, freq, T, d] -> [B, T, freq*d]: full per-frame lag x channel feature.
        h = F.normalize(x, dim=-1).permute(0, 2, 1, 3).reshape(B, T, Fq * d)
        h = self.head(h)
        doa_vec = self.doa(h)
        # Map raw log_std linearly through tanh into [log_std_min, log_std_max].
        raw = self.log_std(h)
        log_std = self.log_std_min + (self.log_std_max - self.log_std_min) * 0.5 * (
            1.0 + torch.tanh(raw)
        )
        return F.normalize(doa_vec, dim=-1), log_std

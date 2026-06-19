"""RTF-MAMBA: a lightweight peak-feature model over the cached REIR.

The cached "rtf" tensor is already the time-domain REIR (``estimate_rtf`` ends
with an ``irfft`` + ``ifftshift``), shaped ``[M-1, T, L]`` with the zero-lag tap
centred. This model takes the per-frame, per-relative-channel peak lag — refined
to sub-sample accuracy by parabolic or sinc interpolation (``cfg.rtf_mamba.interp``)
— as its only feature, and **applies Mamba in the last layer** over the time
axis before the shared ``(doa, log_std)`` readout. Far cheaper than DOAMAMBA: no
TF/channel Mamba stack, just one temporal scan on an ``(M-1)``-wide feature.

Peak extraction is a fixed (non-learned) transform, so it is detached — the
learnable part is the temporal Mamba + head on top of the peak tracks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from mamba_ssm import Mamba

from util import subsample_peak
from .base import DOABase
from .losses import gaussian_loss, von_mises_loss


class RtfMamba(DOABase):
    name = "rtf_mamba"

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.strict_loading = False

        channels = int(getattr(cfg, "receivers_num", None) or getattr(cfg, "recivers_num")) - 1
        self.n_channels = channels

        sub = getattr(cfg, "rtf_mamba", None)
        self.interp = str(getattr(sub, "interp", "parabolic")) if sub is not None else "parabolic"
        self.sinc_os = int(getattr(sub, "sinc_os", 8)) if sub is not None else 8
        expand = int(getattr(sub, "expand", 2)) if sub is not None else 2

        self.negative_log_likelihood_func = (
            gaussian_loss
            if cfg.nnl_func == "gaussian"
            else von_mises_loss if cfg.nnl_func == "von_mises" else None
        )

        # Mamba in the last layer, over time, on the (M-1)-wide peak feature.
        self.mamba = Mamba(
            d_model=channels,
            d_state=cfg.hidden_dim,
            d_conv=cfg.conv_dim,
            expand=expand,
        )

        head_hidden = int(getattr(cfg, "head_hidden", 256))
        self.head = nn.Sequential(
            nn.Linear(channels, head_hidden),
            nn.GELU(),
            nn.Dropout(float(getattr(cfg, "head_dropout", 0.1))),
        )
        self.doa = nn.Sequential(nn.Linear(head_hidden, 2), nn.Tanh())
        self.log_std = nn.Linear(head_hidden, 1)
        self.log_std_min = float(getattr(cfg, "log_std_min", -4.0))
        self.log_std_max = float(getattr(cfg, "log_std_max", 1.5))

        init.xavier_uniform_(self.head[0].weight)
        init.zeros_(self.head[0].bias)

    def _peak_feature(self, x):
        """[B, M-1, T, L] REIR -> [B, T, M-1] normalised sub-sample peak lags."""
        L = x.shape[-1]
        pos = subsample_peak(x.abs(), method=self.interp, os=self.sinc_os)  # [B, M-1, T]
        center = (L - 1) / 2.0
        feat = (pos - center) / center                                      # ~[-1, 1]
        return feat.permute(0, 2, 1).contiguous()                           # [B, T, M-1]

    def forward(self, x):
        feat = self._peak_feature(x).detach()
        h = self.mamba(feat)                # temporal Mamba: [B, T, M-1]
        h = self.head(h)
        doa_vec = self.doa(h)
        raw = self.log_std(h)
        log_std = self.log_std_min + (self.log_std_max - self.log_std_min) * 0.5 * (
            1.0 + torch.tanh(raw)
        )
        return F.normalize(doa_vec, dim=-1), log_std

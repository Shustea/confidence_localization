"""Reusable Mamba residual blocks shared by every DOA model.

Extracted verbatim from the original ``train.py`` so the DOAMAMBA backbone is
byte-for-byte identical (old checkpoints still load). Each block wraps a
``mamba_ssm.Mamba`` scan over one axis of a ``[B, Freq, T, d]`` feature tensor
with a residual connection; ``MambaResCTF`` is the composite block the models
stack.
"""

import torch.nn as nn
from mamba_ssm import Mamba


class MambaResFreq(nn.Module):
    def __init__(self, cfg, expand=2):
        super().__init__()
        self.mamba = Mamba(
            d_model=cfg.d_model,
            d_state=cfg.hidden_dim,
            d_conv=cfg.conv_dim,
            expand=expand,
        )

    def forward(self, x):
        B, Freq, T, d = x.shape
        y = x.contiguous().view(B * Freq, T, d)
        y = self.mamba(y).view(B, Freq, T, d)
        return x + y


class MambaResTime(nn.Module):
    def __init__(self, cfg, expand=2):
        super().__init__()
        self.mamba = Mamba(
            d_model=cfg.d_model,
            d_state=cfg.hidden_dim,
            d_conv=cfg.conv_dim,
            expand=expand,
        )

    def forward(self, x):
        B, Freq, T, d = x.shape
        y = x.transpose(1, 2).contiguous().view(B * T, Freq, d)
        y = self.mamba(y).view(B, T, Freq, d).transpose(1, 2)
        return x + y


class MambaResChannel(nn.Module):
    def __init__(self, cfg, expand: int = 2):
        super().__init__()
        channels = (
            getattr(cfg, "receivers_num", None) or getattr(cfg, "recivers_num", None)
        ) - 1
        d_model = getattr(cfg, "d_model", channels)
        self.channels, self.d_model = channels, d_model
        self.use_proj = channels != d_model
        if self.use_proj:
            self.in_proj = nn.Linear(channels, d_model)
            self.out_proj = nn.Linear(d_model, channels)
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=getattr(cfg, "hidden_dim", 64),
            d_conv=getattr(cfg, "conv_dim", 4),
            expand=expand,
        )
        self.dropout = nn.Dropout(getattr(cfg, "dropout", 0.0))

    def forward(self, x):
        B, F, T, C = x.shape
        y = x.contiguous().view(B, F * T, C)
        if self.use_proj:
            y = self.in_proj(y)
        y = self.mamba(self.norm(y))
        if self.use_proj:
            y = self.out_proj(y)
        y = self.dropout(y).view(B, F, T, C)
        return x + y


class MambaResTF(nn.Module):
    def __init__(self, cfg, expand=2):
        super().__init__()
        self.mambaT = MambaResTime(cfg, expand)
        self.mambaF = MambaResFreq(cfg, expand)

    def forward(self, x):
        return self.mambaF(self.mambaT(x))


class MambaResCTF(nn.Module):
    def __init__(self, cfg, expand=2):
        super().__init__()
        self.mambaTF = MambaResTF(cfg, expand)
        self.mambaC = MambaResChannel(cfg, expand)

    def forward(self, x):
        return self.mambaC(self.mambaTF(x))

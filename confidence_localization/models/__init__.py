"""DOA model package + registry.

``build_model(cfg)`` selects a model by ``cfg.model_type`` (default
``doamamba``), so every consumer — train.py, eval_cached.py, the benchmarks —
swaps architectures through one knob instead of hardcoding a class.
"""

from .base import DOABase
from .blocks import (
    MambaResCTF,
    MambaResChannel,
    MambaResFreq,
    MambaResTF,
    MambaResTime,
)
from .doamamba import DOAMAMBA
from .gcc_mamba import GccKalman, GccMamba, GccMed
from .losses import gaussian_loss, von_mises_loss
from .rtf_mamba import RtfMamba

MODEL_REGISTRY = {
    "doamamba": DOAMAMBA,
    "gcc_mamba": GccMamba,
    "gcc_med": GccMed,
    "gcc_kalman": GccKalman,
    "rtf_mamba": RtfMamba,
}


def build_model(cfg):
    """Instantiate the model named by ``cfg.model_type`` (default 'doamamba')."""
    model_type = str(getattr(cfg, "model_type", "doamamba")).lower()
    if model_type not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model_type '{model_type}'. "
            f"Available: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[model_type](cfg)


__all__ = [
    "DOABase",
    "DOAMAMBA",
    "GccMamba",
    "GccMed",
    "GccKalman",
    "RtfMamba",
    "MambaResCTF",
    "MambaResChannel",
    "MambaResFreq",
    "MambaResTF",
    "MambaResTime",
    "gaussian_loss",
    "von_mises_loss",
    "MODEL_REGISTRY",
    "build_model",
]

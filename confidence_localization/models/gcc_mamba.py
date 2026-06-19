"""GCC-MAMBA family: Mamba over framed GCC-PHAT cross-correlations.

``GccMamba`` reuses the unchanged DOAMAMBA backbone/head by feeding it a derived
config: the ``P = C*(C-1)/2`` mic-pair axis plays the role of the channel/``d_model``
dimension and the cropped lag axis plays the role of ``freq``. So the input
feature ``[B, P, T, n_lags]`` flows through the same ``MambaResCTF`` stack as the
RTF model with no block changes.

``GccMed`` / ``GccKalman`` are the *same model* — they only post-process the
predicted DOA track at inference: a sliding-window median (``GccMed``) or a
constant-velocity Kalman filter (``GccKalman``). Training is untouched (the
smoothing is skipped while ``self.training`` is True) so gradients flow through
``GccMamba`` exactly as normal.
"""

from omegaconf import OmegaConf

from util import kalman_smooth_doa, median_smooth_doa
from .doamamba import DOAMAMBA


def _n_pairs(channels: int) -> int:
    return channels * (channels - 1) // 2


class GccMamba(DOAMAMBA):
    name = "gcc_mamba"

    def __init__(self, cfg):
        super().__init__(self._derive_cfg(cfg))
        # Keep a handle on the user's original config for the smoothing subclasses.
        self.user_cfg = cfg

    @staticmethod
    def _derive_cfg(cfg):
        """Rewrite d_model/freq_dim/recivers_num so DOAMAMBA's blocks fit the GCC
        feature: channel axis = mic pairs (P), freq axis = lag window (n_lags)."""
        channels = int(getattr(cfg, "receivers_num", None) or getattr(cfg, "recivers_num"))
        n_pairs = _n_pairs(channels)
        sub = getattr(cfg, "gcc_mamba", None)
        n_lags = int(getattr(sub, "n_lags", 41)) if sub is not None else 41
        n_lags |= 1
        layers = list(getattr(sub, "layers", cfg.layers)) if sub is not None else list(cfg.layers)

        g = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        g.d_model = n_pairs            # MambaResChannel d_model == channels => no proj
        g.recivers_num = n_pairs + 1   # MambaResChannel: channels = recivers_num - 1
        g.freq_dim = n_lags
        g.layers = layers
        return g


class GccMed(GccMamba):
    name = "gcc_med"

    def __init__(self, cfg):
        super().__init__(cfg)
        sub = getattr(cfg, "gcc_med", None)
        self.med_window = int(getattr(sub, "window", 5)) if sub is not None else 5

    def forward(self, x):
        doa_vec, log_std = super().forward(x)
        if not self.training:
            doa_vec = median_smooth_doa(doa_vec, self.med_window)
        return doa_vec, log_std


class GccKalman(GccMamba):
    name = "gcc_kalman"

    def __init__(self, cfg):
        super().__init__(cfg)
        sub = getattr(cfg, "gcc_kalman", None)
        self.kf_q = float(getattr(sub, "q", 1e-3)) if sub is not None else 1e-3
        self.kf_r = float(getattr(sub, "r", 5e-2)) if sub is not None else 5e-2

    def forward(self, x):
        doa_vec, log_std = super().forward(x)
        if not self.training:
            doa_vec = kalman_smooth_doa(doa_vec, self.kf_q, self.kf_r)
        return doa_vec, log_std

"""LOCATA benchmark harness for DOA-localization algorithms.

Wrap any localization algorithm by subclassing ``LocalizationAlgorithm`` and
pointing ``--algo`` at ``module.path:ClassName[:k=v,...]``. The harness walks
the LOCATA dev (or eval) split, runs the algorithm on each single-source
recording, and compares per-frame azimuth estimates to ground truth.

Splits (single-source recordings only — multi-source tasks 2/4/6 are skipped):
  static : task1            (single static loudspeaker, static array)
  moving : task3 + task5    (single moving talker; static + moving array)

Outputs:
  <out>/per_recording.csv   one row per (task, recording, array)
  <out>/aggregate.csv       MAE / RMSE / acc@10° / acc@15°, frame-weighted
                            within each group (static, moving, all).

Run:
  python confidence_localization/locata_benchmark.py \
      --locata-root data/lab/data/raw/locata \
      --algo confidence_localization.locata_benchmark:ZeroBaseline
"""

from __future__ import annotations

import argparse
import csv
import importlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

# Allow running as a script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.eval_utils import (  # noqa: E402
    build_locata_labels,
    load_locata_waveform,
    preprocess_waveform,
    wrap_radians,
)


# Channel preset matches config.yaml (5 channels picked from the 32-cap eigenmike).
# ``eigenmike_cfg_cross`` is a geometry-matched alternative: capsules selected
# by nearest-unit-direction to the cfg ``receivers_coords`` "+" cross
# (center=apex/+z, +x, +y, -x, -y).  Use it with the classical baselines; the
# DOAMAMBA checkpoint stays on its trained ``eigenmike`` preset.
CFG_CROSS_EIGENMIKE_CHANNELS = (28, 22, 0, 26, 18)
DEFAULT_ARRAY_CHANNELS = {
    "eigenmike": [4, 10, 6, 1, 19],
    "eigenmike_cfg_cross": list(CFG_CROSS_EIGENMIKE_CHANNELS),
    "dicit": list(range(15)),
    "benchmark2": list(range(12)),
    "dummy": [0, 1],
}
# Virtual presets that share another array's on-disk directory layout.
_PHYSICAL_ARRAY = {"eigenmike_cfg_cross": "eigenmike"}

STATIC_TASKS = (1,)
MOVING_TASKS = (3, 5)
SINGLE_SOURCE_TASKS = STATIC_TASKS + MOVING_TASKS


class LocalizationAlgorithm:
    """Pluggable DOA-estimator interface used by the LOCATA benchmark.

    Implement ``__call__`` to return one azimuth estimate per output frame.
    Frame timing is reported via ``hop_seconds`` so the harness can resample
    ground truth onto the same grid.
    """

    name: str = "unnamed"

    def set_channels(self, channels: Sequence[int]) -> None:
        """Optional hook: rebuild internal mic_positions for ``channels``.

        Called once per recording by the harness with the channel list it just
        sliced out of the wav, so geometry-aware algos (GCC / SRP-PHAT) keep
        their ``mic_positions`` in sync with the actual loaded channels.
        Default is a no-op for algos that don't depend on geometry.
        """

    def __call__(self, wav: torch.Tensor, fs: int) -> tuple[np.ndarray, float]:
        """Run the algorithm on a multichannel waveform.

        Args:
            wav: float tensor with shape ``[C, T]``.
            fs: sample rate in Hz.

        Returns:
            ``(azimuth_rad, hop_seconds)`` — ``azimuth_rad`` is a 1-D
            float array of length ``N`` (one azimuth per output frame).
        """
        raise NotImplementedError


class ZeroBaseline(LocalizationAlgorithm):
    """Always predicts 0 rad. Use it to sanity-check the harness end to end."""

    name = "zero_baseline"

    def __init__(self, hop_ms: float = 32.0):
        self.hop_seconds = float(hop_ms) / 1000.0

    def __call__(self, wav, fs):
        n_frames = max(1, int(round(wav.shape[-1] / fs / self.hop_seconds)))
        return np.zeros(n_frames, dtype=np.float32), self.hop_seconds


# --- Eigenmike geometry preset (LOCATA intrinsic frame) ----------------------
# 32 capsule positions in meters, 0-indexed, derived from LOCATA's
# position_array_eigenmike.txt (identical across all dev recordings,
# verified to ~1e-16 m). LOCATA convention: x=right, y=front, z=up;
# +y axis points toward mic 1 ("front of array").
# Mean capsule radius = 0.042 m (mh acoustics em32).

_EIGENMIKE_LOCATA_XYZ = np.array([
    (-0.00000, +0.03921, +0.01505),  # ch  0
    (-0.02226, +0.03562, +0.00000),  # ch  1
    (-0.00000, +0.03921, -0.01505),  # ch  2
    (+0.02226, +0.03562, -0.00000),  # ch  3
    (+0.00000, +0.02226, +0.03562),  # ch  4
    (-0.02433, +0.02433, +0.02409),  # ch  5
    (-0.03921, +0.01505, +0.00000),  # ch  6
    (-0.02433, +0.02433, -0.02409),  # ch  7
    (-0.00000, +0.02226, -0.03562),  # ch  8
    (+0.02433, +0.02433, -0.02409),  # ch  9
    (+0.03921, +0.01505, +0.00000),  # ch 10
    (+0.02433, +0.02433, +0.02409),  # ch 11
    (-0.01505, -0.00026, +0.03921),  # ch 12
    (-0.03562, -0.00000, +0.02226),  # ch 13
    (-0.03600, -0.00000, -0.02163),  # ch 14
    (-0.01505, +0.00026, -0.03921),  # ch 15
    (+0.00000, -0.03921, +0.01505),  # ch 16
    (+0.02226, -0.03562, -0.00000),  # ch 17
    (+0.00000, -0.03921, -0.01505),  # ch 18
    (-0.02226, -0.03562, +0.00000),  # ch 19
    (+0.00000, -0.02226, +0.03562),  # ch 20
    (+0.02433, -0.02433, +0.02409),  # ch 21
    (+0.03921, -0.01505, -0.00000),  # ch 22
    (+0.02433, -0.02433, -0.02409),  # ch 23
    (-0.00000, -0.02226, -0.03562),  # ch 24
    (-0.02433, -0.02433, -0.02409),  # ch 25
    (-0.03921, -0.01505, -0.00000),  # ch 26
    (-0.02433, -0.02433, +0.02409),  # ch 27
    (+0.01505, -0.00026, +0.03921),  # ch 28
    (+0.03562, -0.00000, +0.02226),  # ch 29
    (+0.03562, +0.00000, -0.02226),  # ch 30
    (+0.01505, +0.00026, -0.03921),  # ch 31
], dtype=np.float64)


def eigenmike_mic_positions(channels: Sequence[int]) -> np.ndarray:
    """Return ``[len(channels), 3]`` eigenmike capsule positions (m) in LOCATA's intrinsic frame."""
    return _EIGENMIKE_LOCATA_XYZ[list(channels)].copy()


def _resample_torch(wav: torch.Tensor, fs_in: int, fs_out: int) -> torch.Tensor:
    """Polyphase resample a multichannel float tensor [C, T] via scipy."""
    if int(fs_in) == int(fs_out):
        return wav
    from math import gcd
    from scipy.signal import resample_poly  # local import; scipy is in env
    g = gcd(int(fs_in), int(fs_out))
    up, down = int(fs_out) // g, int(fs_in) // g
    arr = wav.detach().cpu().numpy().astype(np.float64)
    out = resample_poly(arr, up, down, axis=-1)
    return torch.from_numpy(out.astype(np.float32))


class _PairwisePHATMixin:
    """Shared scaffolding: framing, PHAT-weighted GCC, optional resampling.

    Sign convention: cc = ifft(X_i * conj(X_j)); centered (fftshifted) cc peaks
    at lag = delta . u / c * fs  where  delta = r_j - r_i and u is the far-field
    direction-of-arrival unit vector. So (delta . u) = c * lag / fs.
    """

    mic_positions: np.ndarray
    frame_seconds: float
    hop_seconds: float
    sound_speed: float
    target_fs: int | None

    def set_channels(self, channels):
        self.mic_positions = eigenmike_mic_positions(channels)

    def _prepare(self, wav, fs):
        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav)
        if self.target_fs is not None and int(fs) != int(self.target_fs):
            wav = _resample_torch(wav.float(), int(fs), int(self.target_fs))
            fs = int(self.target_fs)
        x = wav.detach().cpu().numpy().astype(np.float64)
        C, T = x.shape
        if C != self.mic_positions.shape[0]:
            raise ValueError(
                f"wav has {C} channels but mic_positions has "
                f"{self.mic_positions.shape[0]} entries — set_channels() wiring "
                "is out of sync with the loaded waveform."
            )
        frame_len = max(2, int(round(self.frame_seconds * fs)))
        hop_len = max(1, int(round(self.hop_seconds * fs)))
        n_frames = 1 if T < frame_len else 1 + (T - frame_len) // hop_len
        n_fft = 1
        while n_fft < 2 * frame_len:
            n_fft *= 2
        return x, int(fs), frame_len, hop_len, n_frames, n_fft

    @staticmethod
    def _pairs(C: int):
        pair_iter = [(i, j) for i in range(C) for j in range(i + 1, C)]
        i_idx = np.fromiter((p[0] for p in pair_iter), dtype=np.int64, count=len(pair_iter))
        j_idx = np.fromiter((p[1] for p in pair_iter), dtype=np.int64, count=len(pair_iter))
        return i_idx, j_idx

    @staticmethod
    def _phat_cc(frame: np.ndarray, n_fft: int, i_idx, j_idx, eps: float = 1e-10):
        X = np.fft.rfft(frame, n=n_fft, axis=-1)               # [C, K]
        P = X[i_idx] * np.conj(X[j_idx])                       # [P, K]
        P /= np.abs(P) + eps                                   # PHAT weighting
        return np.fft.fftshift(np.fft.irfft(P, n=n_fft, axis=-1), axes=-1)  # [P, n_fft]


class GCCBaseline(_PairwisePHATMixin, LocalizationAlgorithm):
    """Per-pair GCC-PHAT TDOA -> least-squares 3D unit DOA -> azimuth.

    For each frame: PHAT-weighted GCC for every mic pair, take the integer-lag
    peak (restricted to physically plausible |tau| <= D_max/c) as the pair's
    TDOA, then solve  delta . u = c * tau  in least squares for unit DOA u
    where delta = r_j - r_i. Azimuth = atan2(u_y, u_x).

    Distinct from SRP-PHAT, which evaluates a steered grid of candidate
    directions instead of inverting per-pair peaks. Same array convention as
    SRPPHATBaseline (x = 0° azimuth, y = 90°).
    """

    name = "gcc"

    def __init__(
        self,
        mic_positions: np.ndarray | None = None,
        frame_ms: float = 64.0,
        hop_ms: float = 32.0,
        sound_speed: float = 343.0,
        target_fs: int | None = 16000,
    ):
        if mic_positions is None:
            mic_positions = eigenmike_mic_positions(CFG_CROSS_EIGENMIKE_CHANNELS)
        self.mic_positions = np.asarray(mic_positions, dtype=np.float64)
        self.frame_seconds = float(frame_ms) / 1000.0
        self.hop_seconds = float(hop_ms) / 1000.0
        self.sound_speed = float(sound_speed)
        self.target_fs = int(target_fs) if target_fs else None

    def __call__(self, wav, fs):
        x, fs, frame_len, hop_len, n_frames, n_fft = self._prepare(wav, fs)
        C, T = x.shape
        center = n_fft // 2
        i_idx, j_idx = self._pairs(C)
        delta = self.mic_positions[j_idx] - self.mic_positions[i_idx]   # [P, 3]

        # Physical-plausibility window on peak lag: |tau| <= |delta| / c.
        max_baseline = float(np.linalg.norm(delta, axis=1).max())
        max_lag = int(np.ceil(max_baseline / self.sound_speed * fs)) + 1
        lo = max(0, center - max_lag)
        hi = min(n_fft, center + max_lag + 1)

        window = np.hanning(frame_len).astype(np.float64)
        out = np.zeros(n_frames, dtype=np.float32)

        for k in range(n_frames):
            start = k * hop_len
            if start + frame_len <= T:
                frame = x[:, start:start + frame_len] * window
            else:
                frame = np.zeros((C, frame_len), dtype=np.float64)
                tail = max(0, T - start)
                if tail > 0:
                    frame[:, :tail] = x[:, start:start + tail]
                frame *= window

            cc = self._phat_cc(frame, n_fft, i_idx, j_idx)              # [P, n_fft]
            peak_lag = np.argmax(cc[:, lo:hi], axis=-1) + lo - center   # [P]
            tau = peak_lag.astype(np.float64) / fs                       # [P], s
            u, *_ = np.linalg.lstsq(delta, self.sound_speed * tau, rcond=None)
            n = float(np.linalg.norm(u))
            if n > 1e-12:
                u = u / n
            out[k] = float(np.arctan2(u[1], u[0]))

        return out, self.hop_seconds


class SRPPHATBaseline(_PairwisePHATMixin, LocalizationAlgorithm):
    """Frame-wise SRP-PHAT azimuth via steered GCC-PHAT across all mic pairs.

    For each frame: PHAT-weighted GCC per pair, then for each azimuth on a
    uniform grid (elevation = 0) sum the linearly-interpolated correlation at
    the far-field-predicted lag. argmax of the summed score is the estimate.
    Use this as the reference baseline; GCCBaseline is its peak-pick variant.

    Defaults to the cfg-cross eigenmike subset (``[28, 22, 0, 26, 18]``);
    for other arrays pass ``mic_positions`` as ``[C, 3]`` Cartesian meters in
    the array's intrinsic frame (x = +0° azimuth, y = +90°).
    """

    name = "srp_phat"

    def __init__(
        self,
        mic_positions: np.ndarray | None = None,
        frame_ms: float = 64.0,
        hop_ms: float = 32.0,
        n_azimuth: int = 360,
        sound_speed: float = 343.0,
        target_fs: int | None = 16000,
    ):
        if mic_positions is None:
            mic_positions = eigenmike_mic_positions(CFG_CROSS_EIGENMIKE_CHANNELS)
        self.mic_positions = np.asarray(mic_positions, dtype=np.float64)
        self.frame_seconds = float(frame_ms) / 1000.0
        self.hop_seconds = float(hop_ms) / 1000.0
        self.n_azimuth = int(n_azimuth)
        self.sound_speed = float(sound_speed)
        self.target_fs = int(target_fs) if target_fs else None

    def __call__(self, wav, fs):
        x, fs, frame_len, hop_len, n_frames, n_fft = self._prepare(wav, fs)
        C, T = x.shape
        center = n_fft // 2
        i_idx, j_idx = self._pairs(C)
        n_pairs = i_idx.size

        azimuths = np.linspace(-np.pi, np.pi, self.n_azimuth, endpoint=False, dtype=np.float64)
        directions = np.stack(
            [np.cos(azimuths), np.sin(azimuths), np.zeros_like(azimuths)], axis=-1,
        )  # [A, 3]

        delta = self.mic_positions[j_idx] - self.mic_positions[i_idx]   # [P, 3]
        # GCC peaks at lag = delta . u / c * fs (centered); evaluate cc at that lag.
        tdoa_seconds = (delta @ directions.T) / self.sound_speed         # [P, A]
        delay_samples = tdoa_seconds * fs + center                       # [P, A]
        lag0 = np.floor(delay_samples).astype(np.int64)
        w1 = (delay_samples - lag0).astype(np.float64)
        w0 = 1.0 - w1
        lag1 = np.clip(lag0 + 1, 0, n_fft - 1)
        lag0 = np.clip(lag0, 0, n_fft - 1)
        rows = np.arange(n_pairs)[:, None]

        window = np.hanning(frame_len).astype(np.float64)
        out = np.zeros(n_frames, dtype=np.float32)

        for k in range(n_frames):
            start = k * hop_len
            if start + frame_len <= T:
                frame = x[:, start:start + frame_len] * window
            else:
                frame = np.zeros((C, frame_len), dtype=np.float64)
                tail = max(0, T - start)
                if tail > 0:
                    frame[:, :tail] = x[:, start:start + tail]
                frame *= window

            cc = self._phat_cc(frame, n_fft, i_idx, j_idx)              # [P, n_fft]
            scores = (w0 * cc[rows, lag0] + w1 * cc[rows, lag1]).sum(axis=0)  # [A]
            out[k] = float(azimuths[int(np.argmax(scores))])

        return out, self.hop_seconds


# Backward-compatible alias: external scripts importing GCCPHATBaseline still work.
GCCPHATBaseline = SRPPHATBaseline


class DOAMambaModel(LocalizationAlgorithm):
    """Wrap a trained DOAMAMBA checkpoint as a LOCATA-benchmark algorithm.

    The waveform is resampled to ``cfg.fs`` (defaults to the training rate), fed
    through the same ``preprocess_waveform`` pipeline used at training time, and
    the model's unit-vector head is converted into a per-frame azimuth in rad.
    """

    name = "doamamba"

    def __init__(
        self,
        checkpoint: str,
        config: str = "config.yaml",
        device: str = "",
        resample: bool = True,
        normalize_rtf: bool = True,
    ):
        from omegaconf import OmegaConf

        repo_root = Path(__file__).resolve().parents[1]
        for p in (repo_root, repo_root / "confidence_localization", repo_root / "data"):
            sp = str(p)
            if sp not in sys.path:
                sys.path.insert(0, sp)
        from train import DOAMAMBA  # noqa: E402

        cfg_path = config
        if not Path(cfg_path).is_absolute():
            cfg_path = str(repo_root / cfg_path)
        self.cfg = OmegaConf.load(cfg_path)

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        state = torch.load(checkpoint, map_location=self.device)
        self.model = DOAMAMBA(self.cfg)
        self.model.load_state_dict(state["state_dict"], strict=False)
        self.model.eval().to(self.device)

        hop_samples = int(self.cfg.nfft * (1 - self.cfg.overlap))
        self.hop_seconds = hop_samples / float(self.cfg.fs)
        self.resample = bool(resample)
        self.normalize_rtf = bool(normalize_rtf)
        self._resampler_key = None
        self._resampler = None

    def _get_resampler(self, fs_in: int, fs_out: int):
        key = (fs_in, fs_out)
        if self._resampler_key != key:
            import torchaudio  # local import to keep top-level light
            self._resampler = torchaudio.transforms.Resample(fs_in, fs_out)
            self._resampler_key = key
        return self._resampler

    def __call__(self, wav: torch.Tensor, fs: int) -> tuple[np.ndarray, float]:
        # STFT helpers build their window on CPU, so the feature pipeline runs
        # CPU-side; the RTF tensor is moved to the model's device afterwards.
        wav = wav.detach().float().cpu()
        target_fs = int(self.cfg.fs)
        if self.resample and fs != target_fs:
            wav = self._get_resampler(int(fs), target_fs)(wav)
            fs_for_rtf = target_fs
        else:
            fs_for_rtf = int(fs)

        rtf = preprocess_waveform(self.cfg, wav, sample_rate=fs_for_rtf, normalize=self.normalize_rtf)
        rtf = rtf.unsqueeze(0).to(self.device)

        with torch.no_grad():
            doa_unit, log_std = self.model(rtf)
        cos = doa_unit[..., 0].squeeze(0).detach().cpu().numpy()
        sin = doa_unit[..., 1].squeeze(0).detach().cpu().numpy()
        azimuth = np.arctan2(sin, cos).astype(np.float32)

        # Cached for the optional debug-plot path below.
        self.last_wav = wav.detach().cpu()
        self.last_fs = int(fs_for_rtf)
        self.last_rtf = rtf.squeeze(0).detach().cpu()
        self.last_log_std = log_std.squeeze(0).squeeze(-1).detach().cpu()
        self.last_doa_unit = doa_unit.squeeze(0).detach().cpu()

        return azimuth, self.hop_seconds


@dataclass
class _RecordingResult:
    task: int
    recording: int
    array: str
    group: str
    n_frames: int
    n_valid: int
    mae_deg: float
    rmse_deg: float
    acc_at_10: float
    acc_at_15: float
    mae_inlier15_deg: float
    n_inlier15: int


def _find_split_root(root: str, split: str) -> Path:
    root_path = Path(root)
    for candidate in (root_path / split, root_path / "LOCATA" / split):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find LOCATA '{split}' under '{root_path}'. "
        "Extract the corresponding archive (dev.zip / eval.zip) first."
    )


def _iter_recordings(
    root: str, split: str, tasks: Sequence[int], array: str
) -> Iterable[tuple[int, int]]:
    base = _find_split_root(root, split)
    for task in tasks:
        task_dir = base / f"task{task}"
        if not task_dir.exists():
            continue
        for rec_dir in sorted(task_dir.iterdir()):
            if not (rec_dir.is_dir() and rec_dir.name.startswith("recording")):
                continue
            if not (rec_dir / array).exists():
                continue
            try:
                rec_idx = int(rec_dir.name.replace("recording", ""))
            except ValueError:
                continue
            yield task, rec_idx


def _angular_error_deg(estimate_rad: np.ndarray, label_rad: np.ndarray) -> np.ndarray:
    diff = wrap_radians(estimate_rad - label_rad)
    return np.rad2deg(np.abs(diff))


def _evaluate_recording(
    algorithm: LocalizationAlgorithm,
    root: str,
    split: str,
    task: int,
    recording: int,
    array: str,
    channels: Sequence[int],
    group: str,
    plots_dir: Path | None = None,
) -> _RecordingResult | None:
    wav, fs, title, _ = load_locata_waveform(root, split, task, recording, array, channels)

    # Consistency checks between wav payload and algorithm expectations.
    if wav.ndim != 2:
        raise ValueError(f"{title}: expected wav [C, T], got shape {tuple(wav.shape)}")
    C_wav, T_wav = wav.shape
    if C_wav != len(channels):
        raise ValueError(
            f"{title}: load_locata_waveform returned {C_wav} channels but "
            f"--channels has {len(channels)} entries: {list(channels)}"
        )
    if not (8000 <= int(fs) <= 192000):
        raise ValueError(f"{title}: implausible sample rate {fs} Hz")
    if T_wav <= 0:
        raise ValueError(f"{title}: empty waveform")
    # Sync geometry-aware baselines to the actually-loaded channel subset.
    algorithm.set_channels(channels)

    estimate, _hop_seconds = algorithm(wav, fs)
    estimate = np.asarray(estimate, dtype=np.float64).reshape(-1)
    n_frames = int(estimate.shape[0])
    if n_frames == 0:
        return None

    labels = build_locata_labels(root, split, task, recording, array, n_frames).numpy()
    valid = np.isfinite(labels)

    if plots_dir is not None:
        try:
            _save_recording_plot(
                plots_dir=plots_dir,
                title=title,
                wav=wav,
                fs=int(fs),
                algorithm=algorithm,
                estimate_rad=estimate,
                labels_rad=labels,
                hop_seconds=float(_hop_seconds),
            )
        except Exception as exc:  # never let plotting crash the run
            print(f"  task{task} rec{recording}: plot failed ({exc})", file=sys.stderr)

    if not valid.any():
        return _RecordingResult(
            task, recording, array, group, n_frames, 0,
            float("nan"), float("nan"), float("nan"), float("nan"),
            float("nan"), 0,
        )

    # Metrics convention:
    #   acc@10°, acc@15°  — fraction of VAD-active frames within threshold (coverage)
    #   MAE, RMSE         — computed on *inliers only* (err < 15°), so catastrophic
    #                       wrong-peak / front-back frames don't drag the localization-
    #                       precision number into orbit. n_inlier15 carries the count
    #                       so a small inlier MAE isn't read without context.
    err = _angular_error_deg(estimate, labels.astype(np.float64))[valid]
    inlier15 = err < 15.0
    n_inlier15 = int(inlier15.sum())
    if n_inlier15 > 0:
        err_in = err[inlier15]
        mae_deg = float(err_in.mean())
        rmse_deg = float(np.sqrt((err_in ** 2).mean()))
    else:
        mae_deg = float("nan")
        rmse_deg = float("nan")
    return _RecordingResult(
        task=task, recording=recording, array=array, group=group,
        n_frames=n_frames, n_valid=int(valid.sum()),
        mae_deg=mae_deg,
        rmse_deg=rmse_deg,
        acc_at_10=float((err <= 10.0).mean()),
        acc_at_15=float((err <= 15.0).mean()),
        mae_inlier15_deg=mae_deg,            # kept as alias for back-compat with CSV reader
        n_inlier15=n_inlier15,
    )


def _save_recording_plot(
    plots_dir: Path,
    title: str,
    wav: torch.Tensor,
    fs: int,
    algorithm: LocalizationAlgorithm,
    estimate_rad: np.ndarray,
    labels_rad: np.ndarray,
    hop_seconds: float,
) -> None:
    """Write a 4-panel PNG: WAV, RTF (if available), DOA + GT (+ ±σ band), per-frame |error|."""
    import matplotlib.pyplot as plt

    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", title)[:120] or "recording"
    plots_dir.mkdir(parents=True, exist_ok=True)

    wav_np = wav.detach().cpu().float().numpy() if torch.is_tensor(wav) else np.asarray(wav)
    if wav_np.ndim != 2:
        wav_np = wav_np.reshape(1, -1)
    ch0 = wav_np[0]
    fs_safe = max(1, int(fs))
    # Trim the noise prefix that estimate_rtf drops so wav and DOA panels share a time axis.
    n_frames_estimate = int(estimate_rad.shape[0])
    wav_duration_s = ch0.shape[0] / fs_safe
    frames_duration_s = max(hop_seconds, n_frames_estimate * float(hop_seconds))
    prefix_s = max(0.0, wav_duration_s - frames_duration_s)
    start_sample = int(round(prefix_s * fs_safe))
    end_sample = start_sample + int(round(frames_duration_s * fs_safe))
    ch0 = ch0[start_sample:end_sample] if end_sample > start_sample else ch0
    t_wav = np.arange(ch0.shape[0]) / fs_safe

    rtf = getattr(algorithm, "last_rtf", None)        # [M-1, T, K] (real)
    log_std = getattr(algorithm, "last_log_std", None)  # [T]
    bound_deg = None
    if log_std is not None:
        bound_rad = log_std.detach().cpu().float().exp().numpy()
        if bound_rad.shape[0] >= estimate_rad.shape[0]:
            bound_rad = bound_rad[: estimate_rad.shape[0]]
        bound_deg = np.rad2deg(bound_rad)

    n_frames = int(estimate_rad.shape[0])
    t_frames = np.arange(n_frames) * float(hop_seconds)
    est_deg = np.rad2deg(estimate_rad)
    label_deg = np.where(np.isfinite(labels_rad), np.rad2deg(labels_rad), np.nan)
    err_rad = ((estimate_rad - labels_rad + np.pi) % (2 * np.pi)) - np.pi
    err_deg = np.where(np.isfinite(labels_rad), np.abs(np.rad2deg(err_rad)), np.nan)

    n_total_frames = n_frames
    n_valid_frames = int(np.isfinite(labels_rad).sum())
    n_inlier15 = int(np.nansum(err_deg <= 15.0))

    has_rtf = rtf is not None
    n_rows = 4 if has_rtf else 3
    height_ratios = [2, 2.5, 3, 2] if has_rtf else [2, 3, 2]

    _BLUE = "#2166ac"
    _RED = "#d6604d"
    rc = {
        "font.family": "serif", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#cccccc",
        "grid.linestyle": "--", "grid.linewidth": 0.6,
        "lines.linewidth": 1.4, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white",
    }

    with plt.rc_context(rc):
        fig, axes = plt.subplots(
            n_rows, 1, figsize=(9, 2.0 * sum(height_ratios) / max(height_ratios)),
            gridspec_kw={"height_ratios": height_ratios},
        )
        fig.suptitle(
            f"{title}  ({n_valid_frames}/{n_total_frames} frames valid · "
            f"{n_inlier15} inlier@15°)",
            fontsize=12, y=0.995,
        )

        # 1) WAV (channel 0).
        ax_wav = axes[0]
        ax_wav.plot(t_wav, ch0, color="#333333", lw=0.6)
        ax_wav.set_ylabel("ch0")
        ax_wav.set_title(f"waveform ch0  (fs={int(fs)} Hz, {ch0.shape[0]} samples)", pad=4)
        ax_wav.set_xlim(t_wav[0], t_wav[-1] if t_wav.size > 1 else 1.0)
        ax_wav.tick_params(labelbottom=False)

        idx = 1

        # 2) RTF slice at mid-time, one line per input channel (M-1 of them).
        if has_rtf:
            ax_rtf = axes[idx]; idx += 1
            rtf_np = rtf.detach().cpu().float().numpy()  # [C, T, K]
            C, T_rtf, L = rtf_np.shape
            t_mid = T_rtf // 2
            lag_axis = np.arange(L) - L // 2
            cmap = plt.get_cmap("viridis")
            for c in range(C):
                ax_rtf.plot(
                    lag_axis, rtf_np[c, t_mid, :],
                    color=cmap(c / max(1, C - 1)),
                    lw=1.0, label=f"ch{c}",
                )
            ax_rtf.axvline(0, color="#888888", lw=0.6, ls=":")
            ax_rtf.set_xlabel("Lag (samples)")
            ax_rtf.set_ylabel("RTF")
            ax_rtf.set_title(
                f"RTF slice @ t = {t_mid}/{T_rtf}  "
                f"({C} input channels, K={L})", pad=4,
            )
            ax_rtf.legend(loc="upper right", framealpha=0.9, edgecolor="#aaaaaa", ncol=C)

        # 3) DOA: estimate + ground truth (+ ±σ band).
        ax_doa = axes[idx]; idx += 1
        if bound_deg is not None:
            ax_doa.fill_between(t_frames, est_deg - bound_deg, est_deg + bound_deg,
                                color=_BLUE, alpha=0.18, label="±1σ", zorder=2)
        ax_doa.plot(t_frames, est_deg, color=_BLUE, lw=1.5, label="estimate", zorder=3)
        ax_doa.plot(t_frames, label_deg, color=_RED, lw=1.5, ls="--", label="ground truth", zorder=4)
        ax_doa.set_ylabel("Azimuth (°)")
        ax_doa.set_title("DOA estimate vs ground truth", pad=4)
        ax_doa.legend(loc="upper right", framealpha=0.9, edgecolor="#aaaaaa")
        ax_doa.tick_params(labelbottom=False)

        # 4) |error| per frame, optionally with the σ bound.
        ax_err = axes[idx]
        if bound_deg is not None:
            ax_err.fill_between(t_frames, 0, bound_deg, color=_BLUE, alpha=0.18, zorder=2)
            ax_err.plot(t_frames, bound_deg, color=_BLUE, lw=1.0, ls="--", label="σ bound", zorder=3)
        ax_err.plot(t_frames, err_deg, color=_RED, lw=1.0, label="|error|", zorder=4)
        ax_err.set_ylabel("Error (°)")
        ax_err.set_xlabel("Time (s)")
        ax_err.set_ylim(bottom=0)
        ax_err.legend(loc="upper right", framealpha=0.9, edgecolor="#aaaaaa")

        # Share the time axis only across frame-indexed panels (WAV + DOA + error).
        # The RTF panel uses lag samples, so it keeps its own xlim.
        time_axes = [axes[0], ax_doa, ax_err]
        for ax in time_axes:
            ax.set_xlim(t_frames[0] if t_frames.size else 0.0,
                        t_frames[-1] if t_frames.size > 1 else 1.0)

        out_path = plots_dir / f"{safe_stem}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)


def _aggregate(results: list[_RecordingResult]) -> dict[str, dict[str, float]]:
    """Frame-weighted aggregate so longer recordings count proportionally."""
    groups: dict[str, list[_RecordingResult]] = {"static": [], "moving": [], "all": []}
    for r in results:
        if r.group in groups:
            groups[r.group].append(r)
        groups["all"].append(r)

    out: dict[str, dict[str, float]] = {}
    for group, items in groups.items():
        if not items:
            continue
        # acc weights = VAD-active frames (denominator of acc@k).
        # MAE / RMSE weights = inlier frames, since per-recording MAE / RMSE are
        # already computed on err < 15° only.
        valid_w = np.array([r.n_valid for r in items], dtype=np.float64)
        inlier_w = np.array([r.n_inlier15 for r in items], dtype=np.float64)
        mae = np.array([r.mae_deg for r in items], dtype=np.float64)
        rmse = np.array([r.rmse_deg for r in items], dtype=np.float64)
        acc10 = np.array([r.acc_at_10 for r in items], dtype=np.float64)
        acc15 = np.array([r.acc_at_15 for r in items], dtype=np.float64)

        acc_mask = valid_w > 0
        if not acc_mask.any():
            continue
        wa = valid_w[acc_mask]

        in_mask = (inlier_w > 0) & np.isfinite(mae)
        if in_mask.any():
            wi = inlier_w[in_mask]
            mae_group = float((mae[in_mask] * wi).sum() / wi.sum())
            rmse_group = float(np.sqrt((rmse[in_mask] ** 2 * wi).sum() / wi.sum()))
            n_inlier_group = int(wi.sum())
        else:
            mae_group = float("nan")
            rmse_group = float("nan")
            n_inlier_group = 0

        out[group] = {
            "n_recordings": int(acc_mask.sum()),
            "n_frames": int(wa.sum()),
            "mae_deg": mae_group,
            "rmse_deg": rmse_group,
            "acc_at_10": float((acc10[acc_mask] * wa).sum() / wa.sum()),
            "acc_at_15": float((acc15[acc_mask] * wa).sum() / wa.sum()),
            "mae_inlier15_deg": mae_group,    # alias retained for back-compat readers
            "n_inlier15": n_inlier_group,
        }
    return out


def _write_per_recording_csv(rows: list[_RecordingResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "group", "task", "recording", "array",
            "n_frames", "n_valid", "mae_deg", "rmse_deg",
            "acc_at_10", "acc_at_15", "mae_inlier15_deg", "n_inlier15",
        ])
        for r in rows:
            w.writerow([
                r.group, r.task, r.recording, r.array,
                r.n_frames, r.n_valid,
                f"{r.mae_deg:.4f}", f"{r.rmse_deg:.4f}",
                f"{r.acc_at_10:.4f}", f"{r.acc_at_15:.4f}",
                f"{r.mae_inlier15_deg:.4f}", r.n_inlier15,
            ])


def _write_aggregate_csv(aggregates: dict[str, dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "n_recordings", "n_frames",
                    "mae_deg", "rmse_deg", "acc_at_10", "acc_at_15",
                    "mae_inlier15_deg", "n_inlier15"])
        for group in ("static", "moving", "all"):
            if group not in aggregates:
                continue
            a = aggregates[group]
            w.writerow([
                group, a["n_recordings"], a["n_frames"],
                f"{a['mae_deg']:.4f}", f"{a['rmse_deg']:.4f}",
                f"{a['acc_at_10']:.4f}", f"{a['acc_at_15']:.4f}",
                f"{a['mae_inlier15_deg']:.4f}", a["n_inlier15"],
            ])


def _coerce(value: str):
    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            pass
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    return value


def _import_algorithm(spec: str) -> LocalizationAlgorithm:
    """Instantiate ``module.path:ClassName[:k=v,k2=v2,...]``."""
    parts = spec.split(":")
    if len(parts) < 2:
        raise ValueError(f"Algorithm spec must be 'module:Class[:k=v,...]', got {spec!r}")
    module_path, class_name = parts[0], parts[1]
    kwargs = {}
    if len(parts) >= 3 and parts[2]:
        for pair in parts[2].split(","):
            if not pair:
                continue
            key, _, value = pair.partition("=")
            kwargs[key.strip()] = _coerce(value.strip())
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    instance = cls(**kwargs)
    if not callable(instance):
        raise TypeError(f"{class_name} is not callable")
    return instance


def _group_for_task(task: int) -> str:
    if task in STATIC_TASKS:
        return "static"
    if task in MOVING_TASKS:
        return "moving"
    return "other"


_DEFAULT_CONFIG = str(Path(__file__).resolve().parents[1] / "config.yaml")


def _load_cfg_if_exists(path: str):
    """Return an OmegaConf cfg if ``path`` exists, else ``None`` (silent)."""
    try:
        from omegaconf import OmegaConf
    except ImportError:
        return None
    cfg_path = Path(path)
    if not cfg_path.is_file():
        return None
    return OmegaConf.load(cfg_path)


def main() -> None:
    p = argparse.ArgumentParser(description="LOCATA benchmark for DOA algorithms.")
    p.add_argument("--config", default=_DEFAULT_CONFIG,
                   help=f"Project config.yaml. Defaults to {_DEFAULT_CONFIG}.")
    p.add_argument("--locata-root", default=None,
                   help="Path to LOCATA raw root (parent of dev/ or LOCATA/dev/). "
                        "Defaults to cfg.locata_root.")
    p.add_argument("--algo", default=None,
                   help="Algorithm spec 'module:Class[:k=v,...]'. If omitted, runs the "
                        "DOAMAMBA model at --checkpoint (or cfg.resume_from_checkpoint).")
    p.add_argument("--checkpoint", default=None,
                   help="DOAMAMBA checkpoint .ckpt path. Defaults to cfg.resume_from_checkpoint.")
    p.add_argument("--array", default="eigenmike",
                   choices=tuple(DEFAULT_ARRAY_CHANNELS.keys()),
                   help="Physical or virtual array preset. 'eigenmike_cfg_cross' "
                        "reads the 'eigenmike' directory but slices the 5 capsules "
                        "(28, 22, 0, 26, 18) that match cfg.receivers_coords.")
    p.add_argument("--channels", default=None,
                   help="Comma-separated 0-indexed channels (default: per-array preset).")
    p.add_argument("--split", default="dev", choices=("dev", "eval"))
    p.add_argument("--include-tasks", default=None,
                   help="Comma-separated task ids to include "
                        f"(default: {','.join(map(str, SINGLE_SOURCE_TASKS))}).")
    p.add_argument("--out", default="locata_benchmark",
                   help="Output directory for CSV results.")
    p.add_argument("--plots-dir", default=None,
                   help="Directory for per-recording PNGs (default: <out>/plots).")
    args = p.parse_args()

    cfg = _load_cfg_if_exists(args.config)

    # Resolve --locata-root from cfg if unset.
    if args.locata_root is None:
        if cfg is None or "locata_root" not in cfg:
            p.error("--locata-root is required (and config.yaml has no 'locata_root').")
        args.locata_root = str(cfg.locata_root)

    # Resolve --algo: explicit > --checkpoint > cfg.resume_from_checkpoint > zero baseline.
    if args.algo is None:
        ckpt = args.checkpoint
        if ckpt is None and cfg is not None:
            raw = cfg.get("resume_from_checkpoint", None) if hasattr(cfg, "get") else None
            if raw and not isinstance(raw, bool):
                ckpt = str(raw)
        if ckpt:
            if not Path(ckpt).is_file():
                p.error(f"checkpoint not found: {ckpt}")
            args.algo = (
                "confidence_localization.locata_benchmark:DOAMambaModel:"
                f"checkpoint={ckpt},config={args.config}"
            )
            print(f"[locata] using DOAMAMBA checkpoint: {ckpt}")
        else:
            args.algo = "confidence_localization.locata_benchmark:ZeroBaseline"
            print("[locata] no --algo / --checkpoint / cfg.resume_from_checkpoint set — "
                  "falling back to ZeroBaseline.", file=sys.stderr)

    if args.channels:
        channels = [int(c) for c in args.channels.split(",") if c.strip()]
    else:
        channels = DEFAULT_ARRAY_CHANNELS.get(args.array, [0, 1])

    if args.include_tasks:
        tasks = tuple(int(t) for t in args.include_tasks.split(","))
        skipped = [t for t in tasks if t not in SINGLE_SOURCE_TASKS]
        if skipped:
            print(f"WARNING: tasks {skipped} are multi-source — first source only.",
                  file=sys.stderr)
    else:
        tasks = SINGLE_SOURCE_TASKS

    algo = _import_algorithm(args.algo)
    label = getattr(algo, "name", args.algo)
    physical_array = _PHYSICAL_ARRAY.get(args.array, args.array)
    print(f"algorithm={label}  array={args.array} (physical={physical_array})  "
          f"channels={channels}  split={args.split}")
    print(f"tasks={tasks}  static={STATIC_TASKS}  moving={MOVING_TASKS}")

    plots_dir = Path(args.plots_dir) if args.plots_dir else Path(args.out) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"[locata] per-recording PNGs -> {plots_dir}")

    results: list[_RecordingResult] = []
    for task, rec in _iter_recordings(args.locata_root, args.split, tasks, physical_array):
        group = _group_for_task(task)
        try:
            r = _evaluate_recording(
                algo, args.locata_root, args.split, task, rec, physical_array,
                channels, group, plots_dir=plots_dir,
            )
        except Exception as exc:  # one bad recording shouldn't kill the run
            print(f"  task{task} rec{rec}: SKIP ({exc})", file=sys.stderr)
            continue
        if r is None:
            continue
        results.append(r)
        mae_str = f"{r.mae_deg:5.2f}°" if r.n_inlier15 > 0 else "  n/a"
        rmse_str = f"{r.rmse_deg:5.2f}°" if r.n_inlier15 > 0 else "  n/a"
        print(
            f"  task{task} rec{rec} [{group:>6s}]: "
            f"MAE={mae_str}  RMSE={rmse_str}  (inliers {r.n_inlier15}/{r.n_valid})  "
            f"acc@10={100 * r.acc_at_10:5.1f}%  acc@15={100 * r.acc_at_15:5.1f}%  "
            f"({r.n_valid}/{r.n_frames} frames)"
        )

    if not results:
        print("No recordings matched.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out)
    _write_per_recording_csv(results, out_dir / "per_recording.csv")
    aggregates = _aggregate(results)
    _write_aggregate_csv(aggregates, out_dir / "aggregate.csv")

    print()
    print(f"per-recording: {out_dir / 'per_recording.csv'}")
    print(f"aggregate:     {out_dir / 'aggregate.csv'}")
    for group in ("static", "moving", "all"):
        if group not in aggregates:
            continue
        a = aggregates[group]
        mae_str = f"{a['mae_deg']:5.2f}°" if a["n_inlier15"] > 0 else "  n/a"
        rmse_str = f"{a['rmse_deg']:5.2f}°" if a["n_inlier15"] > 0 else "  n/a"
        print(
            f"  {group:>6s}: MAE={mae_str}  RMSE={rmse_str}  "
            f"({a['n_inlier15']} inliers)  "
            f"acc@10={100 * a['acc_at_10']:5.1f}%  acc@15={100 * a['acc_at_15']:5.1f}%  "
            f"({a['n_recordings']} recs, {a['n_frames']} frames)"
        )


if __name__ == "__main__":
    main()

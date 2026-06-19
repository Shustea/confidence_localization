"""RealMAN benchmark harness for DOA-localization algorithms.

The RealMAN twin of ``locata_benchmark.py``: it walks the RealMAN source-location
metadata, runs a localization algorithm on each recording, and compares per-frame
azimuth estimates to ground truth using the exact same metric / aggregation /
plotting code as the LOCATA harness (imported from ``locata_benchmark``).

Why a separate harness: RealMAN ships a different array (the 32-capsule Westlake
"audiowu" high-resolution array, NOT the eigenmike) and its ground truth lives in
``{split}_{mode}_source_location.csv`` rather than per-recording LOCATA tables. The
classical baselines (GCC / SRP-PHAT) are geometry-aware, so they get RealMAN mic
positions here; the DOAMAMBA model runs its learned RTF front-end unchanged.

Note on data: only the RealMAN *raw* waveforms that are extracted on disk can be
benchmarked (GCC needs the waveform, not the cached RTF). Rows whose audio is not
present are skipped automatically, so pointing this at the partially-downloaded
``train`` split benchmarks exactly the extracted scenes.

Algorithms (``--algo``):
  gcc        — GCC-PHAT TDOA -> least-squares DOA   (RealMAN geometry)
  srp_phat   — steered SRP-PHAT over an azimuth grid (RealMAN geometry)
  doamamba   — trained DOAMAMBA checkpoint (needs --checkpoint)
  zero       — always-0 sanity baseline
  module:Class[:k=v,...]  — any LocalizationAlgorithm, LOCATA-style spec

Outputs (mirrors LOCATA):
  <out>/per_recording.csv   one row per recording (with title + scene)
  <out>/aggregate.csv       MAE / RMSE / acc@10 / acc@15, frame-weighted
  <out>/plots/*.png         per-recording 3/4-panel diagnostic plots

Run:
  python confidence_localization/realman_benchmark.py --algo gcc \
      --mode moving --split train --out runs/realman_bench_gcc
  python confidence_localization/realman_benchmark.py --algo doamamba \
      --checkpoint outputs/.../best-acc10-...ckpt --out runs/realman_bench_v798
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Allow running as a script from the repo root: add repo, package and data dirs.
_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO, _REPO / "confidence_localization", _REPO / "data"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from data.eval_utils import (  # noqa: E402
    build_realman_labels,
    load_realman_metadata,
    load_realman_waveform,
    preprocess_waveform,
    _resolve_realman_archive,
    _resolve_realman_perchannel_base,
)
from confidence_localization_dataloader import _vad_from_waveform  # noqa: E402

# Reuse the LOCATA harness wholesale: algorithm interface, classical baselines,
# the model wrapper, metric dataclass, plotting, aggregation and CSV writers.
from locata_benchmark import (  # noqa: E402
    DOAMambaModel,
    GCCBaseline,
    LocalizationAlgorithm,
    SRPPHATBaseline,
    ZeroBaseline,
    _RecordingResult,
    _aggregate,
    _angular_error_deg,
    _import_algorithm,
    _resolve_benchmark_config,
    _save_recording_plot,
    _write_aggregate_csv,
)


# --- RealMAN array geometry --------------------------------------------------
# 32-capsule "audiowu" high-resolution array of the Westlake audio lab, copied
# verbatim from the official RealMAN SSL baseline (utils_.audiowu_high_array_geometry):
# three concentric 8-mic circles (R, 2R, 3R) + a center mic + x-axis and z-axis
# capsules. RealMAN's angle(°) labels are annotated in THIS frame, so the
# geometry-aware baselines agree with the labels' azimuth convention
# (azimuth = atan2(y, x), x = 0°, y = 90°).
def _circular_geometry(radius: float, mic_num: int) -> np.ndarray:
    angles = np.arange(0, 2 * np.pi, 2 * np.pi / mic_num)
    pos = np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=-1)
    return pos * radius


def realman_array_geometry() -> np.ndarray:
    R = 0.03
    pos = np.zeros((32, 3))
    pos[1:9, :] = _circular_geometry(R, 8)
    pos[9:17, :] = _circular_geometry(R * 2, 8)
    pos[17:25, :] = _circular_geometry(R * 3, 8)
    pos[25, :] = (-R * 4, 0, 0)
    pos[26, :] = (R * 4, 0, 0)
    pos[27, :] = (R * 5, 0, 0)
    L = 0.045
    pos[28, :] = (0, 0, L * 2)
    pos[29, :] = (0, 0, L)
    pos[30, :] = (0, 0, -L)
    pos[31, :] = (0, 0, -L * 2)
    return pos


def realman_mic_positions(channels: Sequence[int]) -> np.ndarray:
    """Return ``[len(channels), 3]`` RealMAN capsule positions (m)."""
    return realman_array_geometry()[list(channels)].copy()


class RealmanGCC(GCCBaseline):
    """GCC-PHAT baseline wired to the RealMAN array geometry."""

    name = "gcc"

    def set_channels(self, channels):
        self.mic_positions = realman_mic_positions(channels)


class RealmanSRP(SRPPHATBaseline):
    """SRP-PHAT baseline wired to the RealMAN array geometry."""

    name = "srp_phat"

    def set_channels(self, channels):
        self.mic_positions = realman_mic_positions(channels)


_DEFAULT_CHANNELS = [0, 3, 5, 7, 1]


# --- v798 compatibility model -------------------------------------------------
# The v798 checkpoint predates commit 32de66d ("Full (lag x channel) output
# head"). Its head collapses the d_model channel axis with a Linear(d_model->1)
# (``hidden``) and then maps the freq feature straight to (doa, log_std) — there
# is no head MLP. The current train.DOAMAMBA can't load it, so this replicates
# the old head while reusing the unchanged MambaResCTF backbone.
class _DOAMambaV798(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from models import MambaResCTF  # backbone is unchanged across the head rewrite
        self.cfg = cfg
        self.mamba_layers = nn.Sequential(*[MambaResCTF(cfg, e) for e in cfg.layers])
        self.hidden = nn.Linear(cfg.d_model, 1)
        self.doa = nn.Sequential(nn.Linear(cfg.input_dim, 2), nn.Tanh())
        self.log_std = nn.Linear(cfg.input_dim, 1)
        self.log_std_min = float(getattr(cfg, "log_std_min", -4.0))
        self.log_std_max = float(getattr(cfg, "log_std_max", 1.5))

    def forward(self, x):
        x = x.permute(0, -1, 2, 1).contiguous()
        for block in self.mamba_layers:
            x = block(F.normalize(x, dim=-1))
        x_hat = F.gelu(self.hidden(F.normalize(x, dim=-1)).squeeze(-1)).permute(0, 2, 1)
        doa_vec = self.doa(x_hat)
        raw = self.log_std(x_hat)
        log_std = self.log_std_min + (self.log_std_max - self.log_std_min) * 0.5 * (
            1.0 + torch.tanh(raw)
        )
        return F.normalize(doa_vec, dim=-1), log_std


class DOAMambaV798Model(LocalizationAlgorithm):
    """Run a pre-32de66d (v798) DOAMAMBA checkpoint as a benchmark algorithm.

    Mirrors locata_benchmark.DOAMambaModel but builds the old head so the
    checkpoint loads, and caches ``last_rtf`` / ``last_log_std`` for the plots.
    """

    name = "doamamba"

    def __init__(self, checkpoint: str, config: str, device: str = "", normalize_rtf: bool = True):
        from omegaconf import OmegaConf
        self.cfg = OmegaConf.load(config)
        self.device = (
            torch.device(device) if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = _DOAMambaV798(self.cfg)
        state = torch.load(checkpoint, map_location=self.device)
        sd = state.get("state_dict", state)
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        print(f"[realman] loaded v798 checkpoint  (missing={len(missing)} unexpected={len(unexpected)})")
        self.model.eval().to(self.device)
        hop_samples = int(self.cfg.nfft * (1 - self.cfg.overlap))
        self.hop_seconds = hop_samples / float(self.cfg.fs)
        self.normalize_rtf = bool(normalize_rtf)

    def __call__(self, wav: torch.Tensor, fs: int):
        wav = wav.detach().float().cpu()
        target_fs = int(self.cfg.fs)
        if int(fs) != target_fs:
            import torchaudio.functional as taF
            wav = taF.resample(wav, int(fs), target_fs)
            fs = target_fs
        rtf = preprocess_waveform(self.cfg, wav, sample_rate=int(fs), normalize=self.normalize_rtf)
        rtf = rtf.unsqueeze(0).to(self.device)
        with torch.no_grad():
            doa_unit, log_std = self.model(rtf)
        cos = doa_unit[..., 0].squeeze(0).cpu().numpy()
        sin = doa_unit[..., 1].squeeze(0).cpu().numpy()
        azimuth = np.arctan2(sin, cos).astype(np.float32)
        self.last_wav = wav.cpu()
        self.last_fs = int(fs)
        self.last_rtf = rtf.squeeze(0).detach().cpu()
        self.last_log_std = log_std.squeeze(0).squeeze(-1).detach().cpu()
        self.last_doa_unit = doa_unit.squeeze(0).detach().cpu()
        return azimuth, self.hop_seconds


class GccMambaModel(LocalizationAlgorithm):
    """Run a trained ``gcc_*`` checkpoint straight from raw waveforms.

    The GCC models (``gcc_mamba`` / ``gcc_med`` / ``gcc_kalman``) consume the
    cropped GCC-PHAT feature, not the RTF. This computes the exact same feature
    as the GCC cache builder (``util.gcc_phat_frames`` -> ``[P, T, n_lags]``) and
    runs the model, so the held-out RealMAN test set can be benchmarked from the
    raw wavs with no GCC cache to build. ``build_model`` selects gcc_mamba vs the
    median/Kalman-smoothed variants from ``cfg.model_type``.
    """

    name = "gcc_mamba"

    def __init__(self, checkpoint: str, config: str, device: str = ""):
        from omegaconf import OmegaConf
        from models import build_model
        from util import gcc_phat_frames
        self.cfg = OmegaConf.load(config)
        self._gcc_phat_frames = gcc_phat_frames
        self.device = (
            torch.device(device) if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = build_model(self.cfg)
        state = torch.load(checkpoint, map_location=self.device)
        sd = state.get("state_dict", state)
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        print(f"[realman] loaded gcc_mamba checkpoint  (missing={len(missing)} unexpected={len(unexpected)})")
        self.model.eval().to(self.device)
        hop_samples = int(self.cfg.nfft * (1 - self.cfg.overlap))
        self.hop_seconds = hop_samples / float(self.cfg.fs)

    def __call__(self, wav: torch.Tensor, fs: int):
        wav = wav.detach().float().cpu()
        target_fs = int(self.cfg.fs)
        if int(fs) != target_fs:
            import torchaudio.functional as taF
            wav = taF.resample(wav, int(fs), target_fs)
            fs = target_fs
        gcc = self._gcc_phat_frames(wav, self.cfg)         # [P, T, n_lags]
        feat = gcc.unsqueeze(0).to(self.device)            # [1, P, T, n_lags]
        with torch.no_grad():
            doa_unit, log_std = self.model(feat)
        cos = doa_unit[..., 0].squeeze(0).cpu().numpy()
        sin = doa_unit[..., 1].squeeze(0).cpu().numpy()
        azimuth = np.arctan2(sin, cos).astype(np.float32)
        self.last_wav = wav.cpu()
        self.last_fs = int(fs)
        self.last_rtf = gcc.detach().cpu()                 # GCC feature for the plot panel
        self.last_log_std = log_std.squeeze(0).squeeze(-1).detach().cpu()
        self.last_doa_unit = doa_unit.squeeze(0).detach().cpu()
        return azimuth, self.hop_seconds


def _build_doamamba(args, config_path: str):
    """Pick the current vs v798 head by inspecting the checkpoint's keys."""
    if not args.checkpoint:
        raise SystemExit("--algo doamamba requires --checkpoint")
    sd = torch.load(args.checkpoint, map_location="cpu")
    sd = sd.get("state_dict", sd)
    is_v798 = any(k.startswith("hidden.") for k in sd) and not any(k.startswith("head.") for k in sd)
    if is_v798:
        print("[realman] checkpoint has the pre-32de66d (v798) head -> compat model")
        return DOAMambaV798Model(checkpoint=args.checkpoint, config=config_path, device=args.device)
    return DOAMambaModel(checkpoint=args.checkpoint, config=config_path, device=args.device)


def _build_learned(args, config_path: str):
    """Route a learned checkpoint to the right front-end via ``cfg.model_type``.

    GCC models get the GCC-PHAT front-end; everything else gets the RTF one.
    """
    if not args.checkpoint:
        raise SystemExit("--algo doamamba/gcc_mamba requires --checkpoint")
    from omegaconf import OmegaConf
    model_type = str(getattr(OmegaConf.load(config_path), "model_type", "doamamba")).lower()
    if model_type.startswith("gcc"):
        print(f"[realman] model_type={model_type} -> GCC-PHAT front-end (raw-wav)")
        return GccMambaModel(checkpoint=args.checkpoint, config=config_path, device=args.device)
    return _build_doamamba(args, config_path)


def _build_algorithm(args, config_path: str):
    """Resolve ``--algo`` (short name or LOCATA module:Class spec) to an instance."""
    spec = args.algo
    short = {
        "gcc": RealmanGCC,
        "srp": RealmanSRP,
        "srp_phat": RealmanSRP,
        "zero": ZeroBaseline,
    }
    if spec in short:
        return short[spec]()
    if spec in ("doamamba", "model", "gcc_mamba", "gcc_med", "gcc_kalman"):
        return _build_learned(args, config_path)
    return _import_algorithm(spec)


def _scene_of(filename: str) -> str:
    """train/ma_speech/<Scene>/<mode>/<spk>/<file>.flac -> <Scene>."""
    parts = Path(str(filename)).parts
    return parts[2] if len(parts) > 2 else "unknown"


def _audio_present(root: str, filename: str, first_channel: int) -> bool:
    """True if the recording is loadable — either extracted per-channel .flac on
    disk, or readable as a member of its scene .rar (read in-place via bsdtar, no
    extraction). Lets the benchmark run the rar-only test partition directly."""
    if _resolve_realman_perchannel_base(root, filename, first_channel) is not None:
        return True
    archive_path, _ = _resolve_realman_archive(root, filename)
    return archive_path is not None


def _iter_rows(cfg, root: str, split: str, mode: str, scenes, every_nth: int, channels):
    """Yield (index, row, scene) for rows whose channel-0 audio is available."""
    frame = load_realman_metadata(root, split, mode)
    records = frame.to_dict("records")
    kept = 0
    for i, row in enumerate(records):
        if i % every_nth != 0:
            continue
        filename = str(row.get("filename", ""))
        scene = _scene_of(filename)
        if scenes and scene not in scenes:
            continue
        if not _audio_present(root, filename, channels[0]):
            continue
        yield kept, row, scene
        kept += 1


def _evaluate_recording(
    algorithm,
    cfg,
    root: str,
    row,
    index: int,
    scene: str,
    channels,
    use_noisy: bool,
    group: str,
    target_fs: int,
    plots_dir: Path | None,
):
    wav, fs, base_title = load_realman_waveform(root, row, channels, use_noisy=use_noisy)
    if wav.ndim != 2 or wav.shape[0] != len(channels):
        raise ValueError(f"{base_title}: expected wav [C={len(channels)}, T], got {tuple(wav.shape)}")
    wav = wav.float()

    # Resample once to the model's rate so the algorithm, the VAD and the RTF
    # lag scale all share one sample rate (matches the cache builder).
    if int(fs) != int(target_fs):
        import torchaudio.functional as taF
        wav = taF.resample(wav, int(fs), int(target_fs))
        fs = int(target_fs)

    algorithm.set_channels(channels)
    estimate, hop_seconds = algorithm(wav, fs)
    estimate = np.asarray(estimate, dtype=np.float64).reshape(-1)
    n_frames = int(estimate.shape[0])
    if n_frames == 0:
        return None

    labels = build_realman_labels(row, n_frames).numpy().astype(np.float64)
    vad = _vad_from_waveform(cfg, wav, int(fs), n_frames)
    vad_np = vad.cpu().numpy().astype(bool) if torch.is_tensor(vad) else np.asarray(vad, bool)
    labels = np.where(vad_np, labels, np.nan)              # VAD-gate via NaN labels
    valid = np.isfinite(labels)

    title = f"{scene}_{base_title}"
    if plots_dir is not None:
        try:
            _save_recording_plot(
                plots_dir=plots_dir, title=title, wav=wav, fs=int(fs),
                algorithm=algorithm, estimate_rad=estimate,
                labels_rad=labels, hop_seconds=float(hop_seconds),
            )
        except Exception as exc:  # never let plotting kill the run
            print(f"  rec{index}: plot failed ({exc})", file=sys.stderr)

    if not valid.any():
        return _RecordingResult(0, index, scene, group, n_frames, 0,
                                float("nan"), float("nan"), float("nan"),
                                float("nan"), float("nan"), 0), title

    err = _angular_error_deg(estimate, labels)[valid]
    inlier15 = err < 15.0
    n_inlier15 = int(inlier15.sum())
    if n_inlier15 > 0:
        err_in = err[inlier15]
        mae_deg = float(err_in.mean())
        rmse_deg = float(np.sqrt((err_in ** 2).mean()))
    else:
        mae_deg = rmse_deg = float("nan")

    result = _RecordingResult(
        task=0, recording=index, array=scene, group=group,
        n_frames=n_frames, n_valid=int(valid.sum()),
        mae_deg=mae_deg, rmse_deg=rmse_deg,
        acc_at_10=float((err <= 10.0).mean()),
        acc_at_15=float((err <= 15.0).mean()),
        mae_inlier15_deg=mae_deg, n_inlier15=n_inlier15,
    )
    return result, title


def _write_per_recording_csv(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "scene", "title", "n_frames", "n_valid",
                    "mae_deg", "rmse_deg", "acc_at_10", "acc_at_15",
                    "mae_inlier15_deg", "n_inlier15"])
        for title, r in rows:
            w.writerow([
                r.group, r.array, title, r.n_frames, r.n_valid,
                f"{r.mae_deg:.4f}", f"{r.rmse_deg:.4f}",
                f"{r.acc_at_10:.4f}", f"{r.acc_at_15:.4f}",
                f"{r.mae_inlier15_deg:.4f}", r.n_inlier15,
            ])


def main() -> None:
    p = argparse.ArgumentParser(description="RealMAN benchmark for DOA algorithms.")
    p.add_argument("--algo", default="gcc",
                   help="gcc | srp_phat | doamamba | gcc_mamba | zero | module:Class[:k=v,...] "
                        "(doamamba/gcc_mamba auto-pick the RTF vs GCC-PHAT front-end from "
                        "the checkpoint's cfg.model_type)")
    p.add_argument("--checkpoint", default=None, help="DOAMAMBA .ckpt (for --algo doamamba).")
    p.add_argument("--config", default=None,
                   help="Project config. Auto-discovered from --checkpoint's run dir "
                        "when omitted; falls back to repo config.yaml.")
    p.add_argument("--rtf-noise-mode", default="energy", choices=("energy", "prefix"),
                   help="Front-end noise estimation for the model. 'energy' (default) "
                        "matches how the RealMAN cache was built.")
    p.add_argument("--realman-root", default=None, help="RealMAN raw root (default cfg.realman_root).")
    p.add_argument("--split", default="train", choices=("train", "val", "test"),
                   help="Metadata split. Only rows with extracted audio are used.")
    p.add_argument("--mode", default="moving", choices=("moving", "static", "both"))
    p.add_argument("--channels", default=None,
                   help="Comma-separated channels (default cfg.data.train.realman.channels).")
    p.add_argument("--scenes", default=None, help="Comma-separated scene filter (e.g. Gym,Library).")
    p.add_argument("--every-nth", type=int, default=1, help="Keep every Nth metadata row.")
    p.add_argument("--max-recordings", type=int, default=None, help="Cap recordings per mode.")
    p.add_argument("--device", default="", help="Torch device for the model (default auto).")
    p.add_argument("--out", default="runs/realman_bench", help="Output directory.")
    p.add_argument("--plots-dir", default=None, help="PNG dir (default <out>/plots).")
    p.add_argument("--no-plots", action="store_true", help="Disable per-recording plots.")
    args = p.parse_args()

    # Resolve config (architecture from the run's frozen config; force energy mode).
    config_path = _resolve_benchmark_config(
        args.checkpoint, args.config, args.out, rtf_noise_mode=args.rtf_noise_mode
    )
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(config_path)

    root = args.realman_root or str(cfg.realman_root)
    target_fs = int(cfg.fs)
    if args.channels:
        channels = [int(c) for c in args.channels.split(",") if c.strip()]
    else:
        try:
            channels = list(cfg.data.train.realman.channels)
        except Exception:
            channels = list(_DEFAULT_CHANNELS)
    scenes = {s.strip() for s in args.scenes.split(",")} if args.scenes else None
    use_noisy = True
    modes = ("static", "moving") if args.mode == "both" else (args.mode,)

    algorithm = _build_algorithm(args, config_path)
    label = getattr(algorithm, "name", args.algo)
    print(f"algorithm={label}  array=audiowu_high  channels={channels}  "
          f"split={args.split}  modes={modes}  noise_mode={args.rtf_noise_mode}")

    plots_dir = None
    if not args.no_plots:
        plots_dir = Path(args.plots_dir) if args.plots_dir else Path(args.out) / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        print(f"[realman] first-recording-per-scene PNGs -> {plots_dir}")

    rows: list[tuple[str, _RecordingResult]] = []
    plotted: set[tuple[str, str]] = set()   # plot only the first recording per (group, scene)
    for mode in modes:
        group = "static" if mode == "static" else "moving"
        n_mode = 0
        for index, row, scene in _iter_rows(cfg, root, args.split, mode, scenes,
                                            max(1, args.every_nth), channels):
            if args.max_recordings is not None and n_mode >= args.max_recordings:
                break
            do_plot = plots_dir is not None and (group, scene) not in plotted
            if do_plot:
                plotted.add((group, scene))
            try:
                out = _evaluate_recording(
                    algorithm, cfg, root, row, index, scene, channels,
                    use_noisy, group, target_fs, plots_dir if do_plot else None,
                )
            except Exception as exc:
                print(f"  {mode} rec{index}: SKIP ({exc})", file=sys.stderr)
                continue
            if out is None:
                continue
            result, title = out
            rows.append((title, result))
            n_mode += 1
            mae_str = f"{result.mae_deg:5.2f}" if result.n_inlier15 > 0 else "  n/a"
            print(f"  [{group:>6s}] {title[:48]:48s}  MAE={mae_str}  "
                  f"acc@10={100 * result.acc_at_10:5.1f}%  acc@15={100 * result.acc_at_15:5.1f}%  "
                  f"({result.n_valid}/{result.n_frames} frames)")
        print(f"[{mode}] {n_mode} recordings benchmarked")

    if not rows:
        print("No RealMAN recordings with extracted audio matched.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out)
    _write_per_recording_csv(rows, out_dir / "per_recording.csv")
    aggregates = _aggregate([r for _, r in rows])
    _write_aggregate_csv(aggregates, out_dir / "aggregate.csv")

    print()
    print(f"per-recording: {out_dir / 'per_recording.csv'}")
    print(f"aggregate:     {out_dir / 'aggregate.csv'}")
    for grp in ("static", "moving", "all"):
        if grp not in aggregates:
            continue
        a = aggregates[grp]
        mae_str = f"{a['mae_deg']:5.2f}°" if a["n_inlier15"] > 0 else "  n/a"
        rmse_str = f"{a['rmse_deg']:5.2f}°" if a["n_inlier15"] > 0 else "  n/a"
        print(f"  {grp:>6s}: MAE={mae_str}  RMSE={rmse_str}  ({a['n_inlier15']} inliers)  "
              f"acc@10={100 * a['acc_at_10']:5.1f}%  acc@15={100 * a['acc_at_15']:5.1f}%  "
              f"({a['n_recordings']} recs, {a['n_frames']} frames)")


if __name__ == "__main__":
    main()

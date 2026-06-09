import copy
import io
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import soundfile as sf
import torch


_REPO_ROOT = Path(__file__).resolve().parents[1]
_PKG_ROOT = _REPO_ROOT / "confidence_localization"
if str(_PKG_ROOT) not in sys.path:
    sys.path.append(str(_PKG_ROOT))

from util import compute_multichannel_stft, estimate_rtf, save_sample_as_image


def _cfg_with_sample_rate(cfg, sample_rate: Optional[int]):
    if sample_rate is None or int(sample_rate) == int(cfg.fs):
        return cfg
    cloned = copy.copy(cfg)
    cloned.fs = int(sample_rate)
    return cloned


def normalize_rtf(rtf: torch.Tensor) -> torch.Tensor:
    return (rtf - rtf.mean(dim=-1, keepdim=True)) / (rtf.std(dim=-1, keepdim=True) + 1e-6)


def preprocess_waveform(cfg, wav: torch.Tensor, sample_rate: Optional[int] = None, normalize: bool = False) -> torch.Tensor:
    if wav.ndim != 2:
        raise ValueError(f"Expected channel-first waveform shaped [C, T], got {tuple(wav.shape)}")

    working_cfg = _cfg_with_sample_rate(cfg, sample_rate)
    stft = compute_multichannel_stft(wav, working_cfg)
    rtf = estimate_rtf(working_cfg, stft)
    return normalize_rtf(rtf) if normalize else rtf


def limit_records(records, every_nth: int = 1, max_items: Optional[int] = None):
    every_nth = max(int(every_nth or 1), 1)
    sliced = records[::every_nth]
    if max_items is not None:
        sliced = sliced[: int(max_items)]
    return sliced


def sanitize_cache_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")[:120] or "sample"


def cache_file_path(cache_root: str, index: int, title: str) -> Path:
    return Path(cache_root) / f"{index:06d}__{sanitize_cache_name(title)}.pt"


def save_cached_sample(out_path: Path, rtf: torch.Tensor, labels: torch.Tensor, title: str, meta=None, vad=None, overwrite: bool = False):
    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rtf": rtf.detach().cpu(),
        "labels": labels.detach().cpu(),
        "title": str(title),
    }
    if meta is not None:
        payload["meta"] = meta
    if vad is not None:
        payload["vad"] = vad.detach().cpu() if torch.is_tensor(vad) else torch.as_tensor(vad)
    torch.save(payload, out_path)
    return True


def list_cache_files(cache_root: str, every_nth: int = 1, max_items: Optional[int] = None):
    root = Path(cache_root)
    if not root.exists():
        raise FileNotFoundError(f"Cache directory '{root}' does not exist.")

    files = sorted(root.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No cached '.pt' files were found under '{root}'.")

    return limit_records(files, every_nth=every_nth, max_items=max_items)


def load_cached_sample(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        rtf = payload["rtf"] if "rtf" in payload else payload.get("features")
        labels = payload.get("labels")
        title = payload.get("title", Path(path).stem)
        vad = payload.get("vad")
    else:
        rtf = payload
        labels = None
        title = Path(path).stem
        vad = None

    if rtf is None:
        raise ValueError(f"Cached sample '{path}' did not contain an 'rtf' tensor.")

    if labels is not None and not torch.is_tensor(labels):
        labels = torch.tensor(labels, dtype=torch.float32)
    if vad is not None and not torch.is_tensor(vad):
        vad = torch.as_tensor(vad)

    return rtf.float(), labels, str(title), vad


def parse_numeric_series(value) -> np.ndarray:
    if value is None:
        return np.empty(0, dtype=np.float32)
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=np.float32)

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return np.empty(0, dtype=np.float32)

    pieces = [piece for piece in re.split(r"[\s,]+", text) if piece]
    return np.asarray([float(piece) for piece in pieces], dtype=np.float32)


def wrap_radians(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return ((values + np.pi) % (2 * np.pi)) - np.pi


def interpolate_angle_series(values, target_len: int, degrees: bool = False) -> torch.Tensor:
    if target_len <= 0:
        return torch.empty(0, dtype=torch.float32)

    series = parse_numeric_series(values)
    if series.size == 0:
        return torch.full((target_len,), torch.nan, dtype=torch.float32)

    if degrees:
        series = np.deg2rad(series)

    if series.size == 1:
        return torch.full((target_len,), float(wrap_radians(series)[0]), dtype=torch.float32)

    src_idx = np.linspace(0.0, 1.0, num=series.size, dtype=np.float32)
    dst_idx = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    interp = np.interp(dst_idx, src_idx, np.unwrap(series.astype(np.float64)))
    return torch.from_numpy(wrap_radians(interp).astype(np.float32))


def save_prediction_plots(
    doa: torch.Tensor,
    labels: torch.Tensor,
    bound: torch.Tensor,
    title: str,
    filename_stem: str,
    output_dir: str,
):
    os.makedirs(output_dir, exist_ok=True)
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", filename_stem)
    save_sample_as_image(doa.cpu(), labels.cpu(), bound.cpu(), f"{safe_stem}.png", title=title, path=output_dir)


def summarize_prediction(model, doa: torch.Tensor, labels: torch.Tensor, bound: torch.Tensor):
    valid = torch.isfinite(labels)
    if not torch.any(valid):
        return {
            "num_valid": 0,
            "coverage": float("nan"),
            "mean_error_deg": float("nan"),
            "q20_deg": float("nan"),
            "q50_deg": float("nan"),
            "q70_deg": float("nan"),
            "q90_deg": float("nan"),
            "q95_deg": float("nan"),
            "acc10": float("nan"),
            "acc15": float("nan"),
            "errors_deg": torch.empty(0, dtype=torch.float32),
        }

    error = model.circ_error(doa[valid] - labels[valid])
    error_deg = torch.rad2deg(error)
    bound_valid = bound[valid]

    return {
        "num_valid": int(valid.sum().item()),
        "coverage": float((error < bound_valid).float().mean().item()),
        "mean_error_deg": float(error_deg.mean().item()),
        "q20_deg": float(error_deg.quantile(0.20).item()),
        "q50_deg": float(error_deg.quantile(0.50).item()),
        "q70_deg": float(error_deg.quantile(0.70).item()),
        "q90_deg": float(error_deg.quantile(0.90).item()),
        "q95_deg": float(error_deg.quantile(0.95).item()),
        "acc10": float((error_deg < 10.0).float().mean().item()),
        "acc15": float((error_deg < 15.0).float().mean().item()),
        "errors_deg": error_deg.detach().cpu(),
    }


def _normalize_path_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _first_existing_path(candidates: Iterable[Path]) -> Optional[Path]:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_realman_metadata(root: str, split: str, mode: str) -> pd.DataFrame:
    root_path = Path(root)
    csv_name = f"{split}_{mode}_source_location.csv"
    csv_path = _first_existing_path(
        [
            root_path / csv_name,
            root_path / split / csv_name,
            root_path / "RealMAN" / csv_name,
            root_path / "RealMAN" / split / csv_name,
        ]
    )
    if csv_path is None:
        raise FileNotFoundError(f"Could not find RealMAN metadata file '{csv_name}' under '{root_path}'.")

    frame = pd.read_csv(csv_path)
    unnamed = [column for column in frame.columns if str(column).startswith("Unnamed")]
    if unnamed:
        frame = frame.drop(columns=unnamed)
    return frame


def _resolve_realman_recording(root: str, relative_file: str, strict: bool = True) -> Optional[Path]:
    root_path = Path(root)
    rel = Path(str(relative_file))
    candidates = [root_path / rel]
    if rel.parts:
        candidates.extend(
            [
                root_path / Path(*rel.parts[1:]),
                root_path / rel.parts[0] / Path(*rel.parts[1:]),
                root_path / "RealMAN" / rel,
                root_path / "RealMAN" / Path(*rel.parts[1:]),
            ]
        )
    match = _first_existing_path(candidates)
    if match is None and strict:
        raise FileNotFoundError(f"Could not resolve RealMAN file '{relative_file}' under '{root_path}'.")
    return match


def _resolve_realman_archive(root: str, relative_file: str):
    root_path = Path(root)
    rel = Path(str(relative_file))
    if len(rel.parts) < 4:
        return None, None

    split, speech_dir, scene = rel.parts[:3]
    archive_name = f"{scene}.rar"
    member_base = Path(*rel.parts[2:])
    archive_candidates = [
        root_path / split / speech_dir / archive_name,
        root_path / speech_dir / archive_name,
        root_path / "RealMAN" / split / speech_dir / archive_name,
        root_path / "RealMAN" / speech_dir / archive_name,
    ]
    return _first_existing_path(archive_candidates), member_base


def _read_realman_archive_audio(archive_path: Path, member_path: Path):
    bsdtar_path = shutil.which("bsdtar")
    if bsdtar_path is None:
        raise FileNotFoundError(
            "RealMAN audio was not extracted and 'bsdtar' is not available to read the local .rar archives. "
            f"Missing archive member: '{member_path.as_posix()}' from '{archive_path}'."
        )

    try:
        result = subprocess.run(
            [bsdtar_path, "-xOf", str(archive_path), member_path.as_posix()],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        raise FileNotFoundError(
            f"Could not read RealMAN archive member '{member_path.as_posix()}' from '{archive_path}'."
        ) from exc

    audio, sample_rate = sf.read(io.BytesIO(result.stdout), dtype="float32", always_2d=True)
    return torch.from_numpy(audio[:, 0]), sample_rate


def _resolve_realman_perchannel_base(root: str, relative: str, first_channel: int) -> Optional[Path]:
    """Return a base path whose ``_CH{first_channel}`` variant exists on disk.

    RealMAN stores one .flac per channel (``<stem>_CH{n}.flac``); there is no
    base file without a channel suffix. So we resolve by probing the first
    channel's file across the same candidate roots ``_resolve_realman_recording``
    uses, then the caller appends ``_CH{n}`` for every requested channel.
    """
    root_path = Path(root)
    rel = Path(str(relative))
    candidates = [root_path / rel]
    if rel.parts:
        candidates.extend([
            root_path / Path(*rel.parts[1:]),
            root_path / rel.parts[0] / Path(*rel.parts[1:]),
            root_path / "RealMAN" / rel,
            root_path / "RealMAN" / Path(*rel.parts[1:]),
        ])
    for base in candidates:
        ch_path = base.with_name(f"{base.stem}_CH{first_channel}{base.suffix}")
        if ch_path.exists():
            return base
    return None


def load_realman_waveform(root: str, row, channels: Sequence[int], use_noisy: Optional[bool] = None):
    relative = str(row["filename"])
    if use_noisy is not None:
        relative = relative.replace("ma_noisy_speech", "ma_noisy_speech" if use_noisy else "ma_speech")

    # Extracted per-channel layout: resolve via the first channel's _CH file
    # (the bare base file never exists for RealMAN).
    base_path = _resolve_realman_perchannel_base(root, relative, channels[0]) if channels else None
    if base_path is None:
        base_path = _resolve_realman_recording(root, relative, strict=False)
    if base_path is not None:
        paths = [base_path.with_name(f"{base_path.stem}_CH{channel}{base_path.suffix}") for channel in channels]

        waves = []
        sample_rate = None
        for path in paths:
            audio, fs = sf.read(path, dtype="float32", always_2d=True)
            waves.append(torch.from_numpy(audio[:, 0]))
            sample_rate = fs if sample_rate is None else sample_rate

        min_len = min(int(wave.numel()) for wave in waves)
        wav = torch.stack([wave[:min_len] for wave in waves], dim=0)
        return wav, sample_rate, base_path.stem

    archive_path, member_base = _resolve_realman_archive(root, relative)
    if archive_path is None or member_base is None:
        raise FileNotFoundError(
            f"Could not resolve RealMAN file '{relative}' under '{Path(root)}'. "
            "Expected either extracted channel files or a matching scene .rar archive."
        )

    member_paths = [member_base.with_name(f"{member_base.stem}_CH{channel}{member_base.suffix}") for channel in channels]
    waves = []
    sample_rate = None
    for member_path in member_paths:
        wave, fs = _read_realman_archive_audio(archive_path, member_path)
        waves.append(wave)
        sample_rate = fs if sample_rate is None else sample_rate

    min_len = min(int(wave.numel()) for wave in waves)
    wav = torch.stack([wave[:min_len] for wave in waves], dim=0)
    return wav, sample_rate, member_base.stem


def build_realman_labels(row, num_frames: int) -> torch.Tensor:
    angle_value = None
    for key in ("angle(°)", "azimuth", "azi", "Azimuth"):
        if key in row and pd.notna(row[key]):
            angle_value = row[key]
            break
    return interpolate_angle_series(angle_value, num_frames, degrees=True)


def _read_locata_table(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, sep=r"\s+", engine="python", comment="#")
    except Exception:
        frame = pd.read_csv(path)

    unnamed = [column for column in frame.columns if str(column).startswith("Unnamed")]
    if unnamed:
        frame = frame.drop(columns=unnamed)
    return frame


def _find_column(frame: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    normalized = {_normalize_path_key(column): column for column in frame.columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def _find_xyz_columns(frame: pd.DataFrame):
    x_col = _find_column(frame, ("x", "xm", "sourcex", "arrayx", "positionx"))
    y_col = _find_column(frame, ("y", "ym", "sourcey", "arrayy", "positiony"))
    z_col = _find_column(frame, ("z", "zm", "sourcez", "arrayz", "positionz"))
    if x_col and y_col and z_col:
        return x_col, y_col, z_col
    raise KeyError(f"Could not infer x/y/z columns from {list(frame.columns)}")


def _extract_time(frame: pd.DataFrame) -> np.ndarray:
    time_col = _find_column(frame, ("time", "timestamp", "frametime", "sampletime", "t"))
    if time_col is not None:
        return frame[time_col].to_numpy(dtype=np.float64)

    hour_col = _find_column(frame, ("hour",))
    minute_col = _find_column(frame, ("minute",))
    second_col = _find_column(frame, ("second",))
    if hour_col and minute_col and second_col:
        return (
            frame[hour_col].to_numpy(dtype=np.float64) * 3600.0
            + frame[minute_col].to_numpy(dtype=np.float64) * 60.0
            + frame[second_col].to_numpy(dtype=np.float64)
        )

    return np.arange(len(frame), dtype=np.float64)


def _extract_positions(frame: pd.DataFrame):
    x_col, y_col, z_col = _find_xyz_columns(frame)
    positions = frame[[x_col, y_col, z_col]].to_numpy(dtype=np.float64)
    times = _extract_time(frame)
    return times, positions


def _extract_yaw(frame: pd.DataFrame) -> Optional[np.ndarray]:
    yaw_col = _find_column(frame, ("yaw", "azimuth", "rotationz", "rz"))
    if yaw_col is not None:
        yaw = frame[yaw_col].to_numpy(dtype=np.float64)
        return np.deg2rad(yaw) if np.nanmax(np.abs(yaw)) > (2 * np.pi + 1) else yaw

    qw = _find_column(frame, ("qw", "quaternionw"))
    qx = _find_column(frame, ("qx", "quaternionx"))
    qy = _find_column(frame, ("qy", "quaterniony"))
    qz = _find_column(frame, ("qz", "quaternionz"))
    if qw and qx and qy and qz:
        w = frame[qw].to_numpy(dtype=np.float64)
        x = frame[qx].to_numpy(dtype=np.float64)
        y = frame[qy].to_numpy(dtype=np.float64)
        z = frame[qz].to_numpy(dtype=np.float64)
        return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    r11 = _find_column(frame, ("r11", "rotation11"))
    r21 = _find_column(frame, ("r21", "rotation21"))
    if r11 and r21:
        return np.arctan2(frame[r21].to_numpy(dtype=np.float64), frame[r11].to_numpy(dtype=np.float64))

    return None


def _interpolate(values: np.ndarray, src_time: np.ndarray, dst_time: np.ndarray) -> np.ndarray:
    if values.ndim == 1:
        return np.interp(dst_time, src_time, values)

    out = np.empty((dst_time.shape[0], values.shape[1]), dtype=np.float64)
    for dim in range(values.shape[1]):
        out[:, dim] = np.interp(dst_time, src_time, values[:, dim])
    return out


def resolve_locata_recording_dir(root: str, split: str, task: int, recording: int, array: str) -> Path:
    root_path = Path(root)
    rec = Path(f"task{task}") / f"recording{recording}" / array
    candidates = [
        root_path / rec,
        root_path / split / rec,
        root_path / "LOCATA" / split / rec,
        root_path / "LOCATA" / "tasks_1_4" / split / rec,
        root_path / "tasks_1_4" / split / rec,
    ]
    match = _first_existing_path(candidates)
    if match is not None:
        return match

    zip_hint = _first_existing_path(
        [
            root_path / "dev.zip",
            root_path / "eval.zip",
            root_path / "LOCATA" / "dev.zip",
            root_path / "LOCATA" / "eval.zip",
        ]
    )
    if zip_hint is not None:
        raise FileNotFoundError(
            f"Found LOCATA archive '{zip_hint}', but not an extracted recording directory for '{rec}'. "
            "Extract the archive before running evaluation."
        )

    raise FileNotFoundError(f"Could not find LOCATA recording directory for '{rec}' under '{root_path}'.")


def load_locata_waveform(root: str, split: str, task: int, recording: int, array: str, channels: Sequence[int]):
    rec_dir = resolve_locata_recording_dir(root, split, task, recording, array)
    wav_path = rec_dir / f"audio_array_{array}.wav"
    audio, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(audio[:, list(channels)].T)
    title = f"locata_task{task}_rec{recording}_{array}"
    return wav, sample_rate, title, rec_dir


def _load_locata_vad(rec_dir: Path, array: str) -> Optional[np.ndarray]:
    vad_files = sorted(rec_dir.glob(f"VAD_{array}_*.txt"))
    if not vad_files:
        return None

    vad_frame = _read_locata_table(vad_files[0])
    numeric = vad_frame.select_dtypes(include=["number"])
    if numeric.empty:
        return None

    values = numeric.to_numpy(dtype=np.float64)
    if values.ndim == 2 and values.shape[1] > 1:
        values = values[:, -1]
    return values.reshape(-1)


def build_locata_labels(
    root: str,
    split: str,
    task: int,
    recording: int,
    array: str,
    num_frames: int,
    source_name: Optional[str] = None,
) -> torch.Tensor:
    rec_dir = resolve_locata_recording_dir(root, split, task, recording, array)
    source_files = sorted(rec_dir.glob("position_source_*.txt"))
    if not source_files:
        raise FileNotFoundError(f"No LOCATA source position files were found under '{rec_dir}'.")

    source_path = next((path for path in source_files if source_name and source_name in path.name), source_files[0])
    source_frame = _read_locata_table(source_path)
    src_time, src_xyz = _extract_positions(source_frame)

    array_frame = None
    array_time = None
    array_path = rec_dir / f"position_array_{array}.txt"
    if array_path.exists():
        array_frame = _read_locata_table(array_path)
        array_time, array_xyz = _extract_positions(array_frame)
        array_xyz = _interpolate(array_xyz, array_time, src_time)
    else:
        array_xyz = np.zeros_like(src_xyz)

    rel = src_xyz - array_xyz
    azimuth = np.arctan2(rel[:, 1], rel[:, 0])

    rotation_path = rec_dir / f"rotation_array_{array}.txt"
    if rotation_path.exists():
        rotation_frame = _read_locata_table(rotation_path)
        yaw = _extract_yaw(rotation_frame)
        if yaw is not None:
            yaw = _interpolate(yaw, _extract_time(rotation_frame), src_time)
            azimuth = wrap_radians(azimuth - yaw)
    elif array_frame is not None:
        # rotation_array_{array}.txt is often absent; the array orientation is
        # embedded in the rotation_11/rotation_21 columns of position_array_{array}.txt
        yaw = _extract_yaw(array_frame)
        if yaw is not None:
            yaw = _interpolate(yaw, array_time, src_time)
            azimuth = wrap_radians(azimuth - yaw)

    labels = interpolate_angle_series(np.rad2deg(azimuth), num_frames, degrees=True)
    vad = _load_locata_vad(rec_dir, array)
    if vad is not None:
        vad_frame = torch.from_numpy(
            np.interp(
                np.linspace(0.0, 1.0, num=num_frames, dtype=np.float32),
                np.linspace(0.0, 1.0, num=vad.shape[0], dtype=np.float32),
                vad.astype(np.float32),
            )
        )
        labels = labels.masked_fill(vad_frame <= 0.5, torch.nan)
    return labels

import math
import os
import re
import sys
from pathlib import Path

import soundfile as sf
import torch
from numpy import argmax, array, concatenate, zeros
from torch.utils.data import DataLoader, Dataset

# Make sibling helper modules importable when this file is loaded as
# `confidence_localization_dataloader` (added by train.py's sys.path tweak).
_REPO_ROOT = Path(__file__).resolve().parents[1]
_PKG_ROOT = _REPO_ROOT / "confidence_localization"
for p in (str(_REPO_ROOT / "data"), str(_PKG_ROOT)):
    if p not in sys.path:
        sys.path.append(p)

from eval_utils import (
    build_locata_labels,
    build_realman_labels,
    limit_records,
    list_cache_files,
    load_cached_sample,
    load_locata_waveform,
    load_realman_metadata,
    load_realman_waveform,
    normalize_rtf,
    preprocess_waveform,
)
from util import energy_vad, gcc_phat_frames, vad_to_rtf_frames


_g_factor_pattern = re.compile(r"-([-\d.]+)_[\d.-]+_[\d.-]+_")
pi = math.pi


def _cfg_get(node, key, default=None):
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    if hasattr(node, "get"):
        value = node.get(key, default)
        return default if value is None else value
    return getattr(node, key, default)


def _apply_feature_pipeline(rtf, transform=None):
    rtf = normalize_rtf(rtf)
    if transform is not None:
        rtf = transform(rtf)
    return rtf


def _vad_from_waveform(cfg, wav, sample_rate, n_rtf_frames):
    """Compute energy VAD on the reference channel and resample to RTF frame rate.

    Returns all-ones when ``cfg.vad`` is unset (preserves the synthetic contract).
    """
    vad_cfg = _cfg_get(cfg, "vad")
    if vad_cfg is None:
        return torch.ones(n_rtf_frames, dtype=torch.bool)

    if isinstance(wav, torch.Tensor):
        ref = wav[0] if wav.ndim > 1 else wav
    else:
        ref = torch.from_numpy(wav[0] if wav.ndim > 1 else wav).float()

    frame_ms = float(_cfg_get(vad_cfg, "frame_ms", 20))
    hop_ms = float(_cfg_get(vad_cfg, "hop_ms", 10))
    alpha = float(_cfg_get(vad_cfg, "alpha", 4))
    hangover_ms = float(_cfg_get(vad_cfg, "hangover_ms", 150))
    win_sec = float(_cfg_get(vad_cfg, "win_sec", 1))

    vad_raw = energy_vad(
        ref,
        fs=sample_rate,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
        alpha=alpha,
        win_sec=win_sec,
        hangover_ms=hangover_ms,
    )
    rtf_hop_samples = int(cfg.nfft * (1 - cfg.overlap))
    return vad_to_rtf_frames(vad_raw, hop_ms, rtf_hop_samples, sample_rate, n_rtf_frames)


class CLDataset(Dataset):
    """Synthetic dataset: pre-extracted .pt RTF tensors + label inference from filename.

    Returns ``(rtf, labels, title, vad, wav_ch0)``. VAD is all-ones for synthetic
    since the speech occupies the full sample (modulo the ``pre_speech_noise_time``
    prelude which is already trimmed inside ``estimate_rtf``). ``wav_ch0`` is the
    channel-0 waveform lazy-loaded from the sibling .wav (zeros of the expected
    length when absent, so collate still stacks deterministically).
    """

    def __init__(self, cfg, root_dir, transform=None):
        self.root_dir = root_dir
        self.cfg = cfg
        self.transform = transform
        self.classification = cfg.classification
        self.sample_files = [
            os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(".pt")
        ]
        # The synthetic pipeline's apply_rir_on_sample trims to sample_length_secs
        # (the noise prefix is folded into the RTF whitening window, not the wav),
        # so the on-disk .wav has this many samples per channel.
        self._expected_wav_len = int(round(float(cfg.fs) * float(cfg.sample_length_secs)))

    def __len__(self):
        return len(self.sample_files)

    def _load_wav_ch0(self, sample_name: str) -> torch.Tensor:
        wav_path = sample_name.replace(".pt", ".wav")
        if os.path.exists(wav_path):
            try:
                audio, _ = sf.read(wav_path, dtype="float32", always_2d=True)
                ch0 = torch.from_numpy(audio[:, 0])
                if ch0.shape[0] >= self._expected_wav_len:
                    return ch0[: self._expected_wav_len]
                pad = self._expected_wav_len - ch0.shape[0]
                return torch.cat([ch0, torch.zeros(pad, dtype=ch0.dtype)])
            except Exception:
                pass
        return torch.zeros(self._expected_wav_len, dtype=torch.float32)

    def __getitem__(self, idx):
        sample_name = self.sample_files[idx]
        rtf = torch.load(sample_name)
        speakers = list(reversed(get_speakers_from_path(sample_name[:-3])))
        title = torch.tensor(
            list(reversed(get_speaker_doa_from_path(sample_name[:-3])[: len(speakers)]))
        )

        labels = assign_gt_to_tf_bin(self.cfg, rtf[0].shape, sample_name, self.classification)
        rtf = _apply_feature_pipeline(rtf, self.transform)
        vad = torch.ones(rtf.shape[1], dtype=torch.bool)
        wav_ch0 = self._load_wav_ch0(sample_name)
        return rtf, labels, str(title)[8:-2], vad, wav_ch0


class CachedExternalDataset(Dataset):
    """Loads pre-cached `.pt` samples (RealMAN / LOCATA preprocessed by make_dataset.py)."""

    def __init__(self, cache_root, transform=None, every_nth=1, max_items=None, target_frames=None):
        self.transform = transform
        self.target_frames = target_frames
        self.cache_files = list_cache_files(cache_root, every_nth=every_nth, max_items=max_items)

    def __len__(self):
        return len(self.cache_files)

    def _pad_or_truncate(self, rtf, labels, vad):
        T = rtf.shape[1]
        target = self.target_frames
        if T > target:
            start = torch.randint(0, T - target, (1,)).item()
            rtf = rtf[:, start:start + target, :]
            labels = labels[start:start + target]
            vad = vad[start:start + target]
        elif T < target:
            pad = target - T
            rtf = torch.cat([rtf, rtf[:, -1:, :].expand(-1, pad, -1)], dim=1)
            labels = torch.cat([labels, torch.full((pad,), torch.nan)])
            vad = torch.cat([vad, torch.zeros(pad, dtype=vad.dtype)])
        return rtf, labels, vad

    def __getitem__(self, idx):
        rtf, labels, title, vad = load_cached_sample(self.cache_files[idx])
        if labels is None:
            raise ValueError(f"Cached sample '{self.cache_files[idx]}' does not include labels.")
        if vad is None:
            vad = torch.ones(rtf.shape[1], dtype=torch.bool)
        else:
            vad = vad.to(torch.bool)
        if self.target_frames is not None:
            rtf, labels, vad = self._pad_or_truncate(rtf, labels, vad)
        # No wav cached for this dataset; return a sentinel.
        wav_ch0 = torch.zeros(1, dtype=torch.float32)
        return _apply_feature_pipeline(rtf, self.transform), labels, title, vad, wav_ch0


class RealMANDataset(Dataset):
    """Realtime RealMAN: load wav, compute STFT+RTF on the fly, energy-VAD from wav."""

    def __init__(self, cfg, dataset_cfg, transform=None):
        self.cfg = cfg
        self.transform = transform
        self.root = _cfg_get(dataset_cfg, "root", _cfg_get(cfg, "realman_root"))
        self.split = _cfg_get(dataset_cfg, "split", "val")
        self.mode = _cfg_get(dataset_cfg, "mode", "moving")
        self.use_noisy = bool(_cfg_get(dataset_cfg, "use_noisy", True))
        self.channels = list(
            _cfg_get(dataset_cfg, "channels", _cfg_get(dataset_cfg, "chs", [0, 1]))
        )
        every_nth = _cfg_get(dataset_cfg, "every_nth", 1)
        max_items = _cfg_get(dataset_cfg, "max_items", None)

        frame = load_realman_metadata(self.root, self.split, self.mode)
        self.rows = limit_records(frame.to_dict("records"), every_nth=every_nth, max_items=max_items)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        wav, sample_rate, title = load_realman_waveform(
            self.root, row, self.channels, use_noisy=self.use_noisy
        )
        rtf = preprocess_waveform(self.cfg, wav, sample_rate=sample_rate, normalize=False)
        labels = build_realman_labels(row, rtf.shape[1])
        vad = _vad_from_waveform(self.cfg, wav, sample_rate, rtf.shape[1])
        wav_ch0 = wav[0].float()
        return _apply_feature_pipeline(rtf, self.transform), labels, title, vad, wav_ch0


class LOCATADataset(Dataset):
    """Realtime LOCATA: load wav, compute STFT+RTF on the fly, energy-VAD from wav."""

    def __init__(self, cfg, dataset_cfg, transform=None):
        self.cfg = cfg
        self.transform = transform
        self.root = _cfg_get(dataset_cfg, "root", _cfg_get(cfg, "locata_root"))
        self.split = _cfg_get(dataset_cfg, "split", "dev")
        self.task = int(_cfg_get(dataset_cfg, "task", 1))
        self.recording = int(_cfg_get(dataset_cfg, "recording", 1))
        self.array = _cfg_get(dataset_cfg, "array", "eigenmike")
        self.channels = list(
            _cfg_get(dataset_cfg, "channels", _cfg_get(dataset_cfg, "chs", [0, 1]))
        )
        self.source_name = _cfg_get(dataset_cfg, "source_name", None)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        del idx
        wav, sample_rate, title, _ = load_locata_waveform(
            self.root,
            self.split,
            self.task,
            self.recording,
            self.array,
            self.channels,
        )
        rtf = preprocess_waveform(self.cfg, wav, sample_rate=sample_rate, normalize=False)
        labels = build_locata_labels(
            self.root,
            self.split,
            self.task,
            self.recording,
            self.array,
            rtf.shape[1],
            source_name=self.source_name,
        )
        vad = _vad_from_waveform(self.cfg, wav, sample_rate, rtf.shape[1])
        wav_ch0 = wav[0].float()
        return _apply_feature_pipeline(rtf, self.transform), labels, title, vad, wav_ch0


class GccDataset(Dataset):
    """Realtime GCC-PHAT over RealMAN wav: load wav, compute framed GCC-PHAT.

    The cached counterpart is just a ``CachedExternalDataset`` pointed at the
    GCC cache (built by ``data/build_gcc_cache_splits.py``), since the GCC
    feature is stored under the same payload key as the RTF one.
    """

    def __init__(self, cfg, dataset_cfg, transform=None):
        self.cfg = cfg
        self.transform = transform
        self.root = _cfg_get(dataset_cfg, "root", _cfg_get(cfg, "realman_root"))
        self.split = _cfg_get(dataset_cfg, "split", "val")
        self.mode = _cfg_get(dataset_cfg, "mode", "moving")
        self.use_noisy = bool(_cfg_get(dataset_cfg, "use_noisy", True))
        self.channels = list(
            _cfg_get(dataset_cfg, "channels", _cfg_get(dataset_cfg, "chs", [0, 1]))
        )
        every_nth = _cfg_get(dataset_cfg, "every_nth", 1)
        max_items = _cfg_get(dataset_cfg, "max_items", None)

        frame = load_realman_metadata(self.root, self.split, self.mode)
        self.rows = limit_records(frame.to_dict("records"), every_nth=every_nth, max_items=max_items)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        wav, sample_rate, title = load_realman_waveform(
            self.root, row, self.channels, use_noisy=self.use_noisy
        )
        wav = wav.float()
        target_fs = int(self.cfg.fs)
        if int(sample_rate) != target_fs:
            import torchaudio.functional as taF
            wav = taF.resample(wav, int(sample_rate), target_fs)
            sample_rate = target_fs
        gcc = gcc_phat_frames(wav, self.cfg)              # [P, T, n_lags]
        labels = build_realman_labels(row, gcc.shape[1])
        vad = _vad_from_waveform(self.cfg, wav, sample_rate, gcc.shape[1])
        wav_ch0 = wav[0].float()
        return _apply_feature_pipeline(gcc, self.transform), labels, title, vad, wav_ch0


def assign_gt_to_tf_bin(cfg, spectrum_shape_tuple, path, classification):
    all_spectra = []
    _ = 0 if classification else 0

    speakers = list(reversed(get_speakers_from_path(path[:-3])))
    T, _ = spectrum_shape_tuple

    labels = torch.full((cfg.max_num_of_speakers, T), torch.nan)
    doas = torch.tensor(list(reversed(get_speaker_doa_from_path(path[:-3])[: len(speakers)])))
    g = list(reversed(get_g_factor_from_path(path[:-3])))

    for speaker_idx, speaker in enumerate(speakers):
        file_path = os.path.join(cfg.wav_path, speaker[:3], f"{speaker}.wav")
        if not os.path.exists(file_path):
            print(f"Missing file: {file_path}")
            continue
        if os.path.getsize(file_path) == 0:
            print(f"Empty file: {file_path}")
            continue

        try:
            signal, _ = sf.read(file_path)
        except Exception as exc:
            print(f"Failed to read {file_path}: {exc}")
            continue

        target_length = cfg.fs * cfg.sample_length_secs
        signal = (
            concatenate([signal, zeros(target_length - len(signal))])
            if len(signal) < target_length
            else signal[:target_length]
        )

        if speaker_idx > 0 and g[speaker_idx] is not None:
            signal = float(g[speaker_idx]) * signal

        signal_power = torch.tensor(signal ** 2)
        all_spectra.append(signal_power.median())

    for speaker_idx in range(len(speakers)):
        pre_to_sig_ratio = cfg.pre_speech_noise_time / cfg.sample_length_secs
        doa_start, doa_end = doas[speaker_idx]
        doa_map = torch.linspace(doa_start, doa_end, int(labels.shape[1] * (1 + pre_to_sig_ratio)))[
            int(labels.shape[1] * pre_to_sig_ratio):
        ]
        labels[speaker_idx] = doa_map

    return torch.remainder(labels[argmax(array(all_spectra))] + pi, 2 * pi) - pi


def get_speaker_positions_from_path(path):
    segments = re.findall(r"(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)_", path.split("/")[-1])
    positions = []
    for match in segments:
        try:
            positions.append([float(x) for x in match])
        except ValueError:
            print(f"Invalid coordinates found: {match}")
    return positions


def get_g_factor_from_path(path: str):
    g_factor = [None]
    match = _g_factor_pattern.search(path)
    if match:
        g_factor.append(float(match.group(1)))
    return g_factor


def get_speakers_from_path(path):
    return re.findall(r"(?<=_)\d+[a-z][a-z0-9]+(?=[-_])", path.split("/")[-1])


def get_speaker_doa_from_path(path):
    return [tuple(float(s) for s in p[:2]) for p in get_speaker_positions_from_path(path)]


def _stage_cfg(cfg, stage):
    data_cfg = _cfg_get(cfg, "data")
    if data_cfg is not None:
        section = _cfg_get(data_cfg, stage)
        if section is not None:
            return section
    return None


def _loader_value(cfg, stage_cfg, key, default):
    loader_cfg = _cfg_get(stage_cfg, "loader")
    if loader_cfg is not None:
        value = _cfg_get(loader_cfg, key)
        if value is not None:
            return value
    return _cfg_get(cfg, key, default)


def _synthetic_root(cfg, stage_cfg, stage, root_dir):
    if root_dir is not None:
        return root_dir

    synthetic_cfg = _cfg_get(stage_cfg, "synthetic")
    if synthetic_cfg is not None:
        root = _cfg_get(synthetic_cfg, "root")
        if root is not None:
            return root

    if stage == "train":
        return _cfg_get(cfg, "train_path")
    if stage in ("val", "eval"):
        return _cfg_get(cfg, "val_path")
    return _cfg_get(cfg, "test_path")


def _feature_mode(dataset_cfg):
    return str(_cfg_get(dataset_cfg, "feature_mode", "realtime")).lower()


def _target_frames(cfg, dataset_cfg):
    dur = _cfg_get(dataset_cfg, "target_duration_sec", None)
    if dur is None:
        return None
    hop = int(_cfg_get(cfg, "nfft", 1024) * (1 - _cfg_get(cfg, "overlap", 0.75)))
    return int(float(dur) * int(_cfg_get(cfg, "fs", 16000)) / hop)


def _external_dataset(cfg, stage_cfg, source, transform=None):
    if source == "gcc":
        dataset_cfg = _cfg_get(stage_cfg, "gcc", stage_cfg)
        default_root = f"{_cfg_get(cfg, 'realman_gcc_target')}/train"
        if _feature_mode(dataset_cfg) in {"cached", "cache", "precomputed"}:
            return CachedExternalDataset(
                _cfg_get(dataset_cfg, "cache_root", default_root),
                transform=transform,
                every_nth=_cfg_get(dataset_cfg, "every_nth", 1),
                max_items=_cfg_get(dataset_cfg, "max_items", None),
                target_frames=_target_frames(cfg, dataset_cfg),
            )
        return GccDataset(cfg, dataset_cfg, transform=transform)

    if source == "realman":
        dataset_cfg = _cfg_get(stage_cfg, "realman", stage_cfg)
        if _feature_mode(dataset_cfg) in {"cached", "cache", "precomputed"}:
            return CachedExternalDataset(
                _cfg_get(dataset_cfg, "cache_root", _cfg_get(cfg, "realman_target")),
                transform=transform,
                every_nth=_cfg_get(dataset_cfg, "every_nth", 1),
                max_items=_cfg_get(dataset_cfg, "max_items", None),
                target_frames=_target_frames(cfg, dataset_cfg),
            )
        return RealMANDataset(cfg, dataset_cfg, transform=transform)

    dataset_cfg = _cfg_get(stage_cfg, "locata", stage_cfg)
    if _feature_mode(dataset_cfg) in {"cached", "cache", "precomputed"}:
        return CachedExternalDataset(
            _cfg_get(dataset_cfg, "cache_root", _cfg_get(cfg, "locata_target")),
            transform=transform,
            every_nth=_cfg_get(dataset_cfg, "every_nth", 1),
            max_items=_cfg_get(dataset_cfg, "max_items", None),
            target_frames=_target_frames(cfg, dataset_cfg),
        )
    return LOCATADataset(cfg, dataset_cfg, transform=transform)


def get_dataloader(cfg, root_dir=None, shuffle=False, transform=None, stage="train"):
    """Build a DataLoader.

    Legacy mode: pass ``root_dir`` (synthetic only) — matches c1ff720's call site.
    Stage mode: omit ``root_dir`` and use ``stage=train|val|eval`` to dispatch via
    ``cfg.data.<stage>.source`` (synthetic | realman | locata).
    """
    stage_cfg = _stage_cfg(cfg, stage)
    source = str(_cfg_get(stage_cfg, "source", "synthetic")).lower()

    if root_dir is None and source in {"realman", "locata", "gcc"}:
        dataset = _external_dataset(cfg, stage_cfg, source, transform=transform)
    else:
        dataset = CLDataset(cfg, _synthetic_root(cfg, stage_cfg, stage, root_dir), transform=transform)

    batch_size = int(_loader_value(cfg, stage_cfg, "batch_size", _cfg_get(cfg, "batch_size", 1)))
    num_workers = int(_loader_value(cfg, stage_cfg, "num_workers", _cfg_get(cfg, "num_workers", 0)))
    pin_memory = bool(_loader_value(cfg, stage_cfg, "pin_memory", True))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
    )

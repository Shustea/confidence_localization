# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sound source localization (Direction of Arrival / DOA estimation) in the time-frequency domain using a Mamba-based neural network, with a calibrated confidence/uncertainty score alongside each prediction. The model outputs a 2D unit vector (DOA) and a Von Mises concentration parameter κ (confidence).

## Commands

```bash
# Install dependencies
make requirements
# or: pip install -r requirements.txt

# Train DOA model (full model, MAE loss)
python confidence_localization/train.py train_mode=doa

# Train confidence head only (freeze DOA, von Mises NLL)
python confidence_localization/train.py train_mode=confidence resume_from_checkpoint=<doa_ckpt>

# Evaluate / test the model
python confidence_localization/test.py

# Preprocess RealMAN data to cached .pt files
python data/make_dataset.py make_dataset.mode=realman

# Preprocess / generate synthetic dataset
python data/make_dataset.py make_dataset.mode=synthetic

# Lint
make lint        # flake8 + isort + black (check only)
make format      # auto-format with black
```

Configuration is managed via Hydra — `config.yaml` is the primary config file. Override any setting on the command line, e.g. `python confidence_localization/train.py batch_size=32`.

## Architecture

### Data Pipeline

Raw audio → `util.compute_multichannel_stft()` → `util.estimate_rtf()` (noise-whitened Relative Transfer Function) → normalized RTF tensors (shape `[freq, time, channels]`) → DOAMAMBA model.

Three dataset sources share the same RTF feature format:
- **Synthetic** (`CLDataset`): pre-computed tensors under `data/processed/`
- **RealMAN** (`RealMANDataset` / `CachedExternalDataset`): real recordings from `data/raw/RealMAN`, cached as `.pt` under `data/lab/processed/.../RealMAN/{train,val}`
- **LOCATA** (`LOCATADataset`): real recordings from `data/raw/locata`

Preprocessing (caching RTFs) is done by `data/make_dataset.py make_dataset.mode=realman`. The `data/eval_utils.py` handles label interpolation, caching, and evaluation summaries for real datasets.

`CachedExternalDataset` supports `target_duration_sec` (set in config per split) which pads/truncates all samples to a fixed frame count. Samples shorter than 2 seconds should be filtered from the cache before training (see data filtering notes below).

### Model (`confidence_localization/train.py`)

`DOAMAMBA` (PyTorch Lightning module):
- **Backbone:** Stacked `MambaResCTF` blocks that mix along frequency, time, and channel axes independently using `CompatibleMamba` (CUDA-optimized with CPU fallback).
- **DOA head:** outputs a 2D unit vector `[B, T, 2]` (sin/cos of azimuth angle).
- **Confidence head:** outputs κ (Von Mises concentration), converted to circular std via `kappa_to_circ_std()`.

Key hyperparameters in `config.yaml`: `d_model`, `hidden_dim`, `layers` (list of expand factors), `input_dim` (= `nfft`), `freq_dim`, `time_dim`.

### Loss Functions (`confidence_localization/model.py`)

Two-stage training with different losses per stage:
- **DOA mode** (`train_mode: doa`): Mean Angular Error (MAE) + temporal smoothness regularizer. No confidence loss — kappa head runs but doesn't affect the loss.
- **Confidence mode** (`train_mode: confidence`): `vm_nll_calibrated()` — Von Mises NLL with a soft coverage penalty to calibrate κ. DOA head is frozen.

Other loss functions available but not currently active: `von_mises_loss()`, `gaussian_loss()`, `hetero_gaussian_nll_err()`, `bound_loss()`, etc.

### Training Details

- PyTorch Lightning with 16-bit mixed precision
- AdamW optimizer with `ReduceLROnPlateau` scheduler
- Three ModelCheckpoint callbacks: best val loss, best accuracy, best calibration
- Resume from checkpoint: set `resume_from_checkpoint` in `config.yaml`
- Two-stage workflow:
  1. Train DOA: `train_mode: doa` — trains full model end-to-end with MAE loss
  2. Train confidence: `train_mode: confidence` — loads DOA checkpoint, freezes DOA head via `train_logstd_only()`, trains only kappa/bound heads with von Mises NLL
- Data source controlled by `data.train.source` / `data.val.source` (synthetic, realman, or locata)
- RealMAN cached data uses `feature_mode: cached` with `target_duration_sec: 3` (pad/truncate to 188 frames)

### Evaluation (`confidence_localization/test.py`)

Four-stage pipeline: prepare dataloader → load checkpoint → run batch inference → aggregate metrics.

Metrics reported: accuracy@10°, accuracy@15°, coverage at predicted bounds, quantile errors (q20/q50/q70/q90/q95). Plots saved to `samples/`.

## Key Config Parameters

```yaml
# Audio
fs: 16000       # sample rate
nfft: 1024      # FFT size (also = input_dim)
overlap: 0.75   # STFT overlap

# Model
d_model: 4      # Mamba hidden state dim
hidden_dim: 64
layers: [2, 4, 4, 4]

# Training
batch_size: 8
lr: 1e-4
epochs: 200
train_mode: doa          # "doa" = MAE loss; "confidence" = freeze DOA, von Mises NLL
temporal_reg_factor: 1e-4

# Data source
data.train.source: realman    # synthetic | realman | locata
data.val.source: realman
target_duration_sec: 3        # pad/truncate cached samples to 3 seconds (188 frames)

# Paths (outputs)
logs: logs/
models: models/
samples: samples/
```

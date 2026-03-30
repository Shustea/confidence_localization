# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sound source localization (Direction of Arrival / DOA estimation) in the time-frequency domain using a Mamba-based neural network, with a calibrated confidence/uncertainty score alongside each prediction. The model outputs a 2D unit vector (DOA) and a Von Mises concentration parameter κ (confidence).

## Commands

```bash
# Install dependencies
make requirements
# or: pip install -r requirements.txt

# Train the model
python confidence_localization/train.py

# Evaluate / test the model
python confidence_localization/test.py

# Preprocess / generate synthetic dataset
python data/make_dataset.py +dataset.mode=generate_synthetic

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
- **RealMAN** (`RealMANDataset`): real recordings from `data/raw/`
- **LOCATA** (`LOCATADataset`): real recordings from `data/raw/locata`

Preprocessing (caching RTFs) is done by `data/make_dataset.py`. The `data/eval_utils.py` handles label interpolation, caching, and evaluation summaries for real datasets.

### Model (`confidence_localization/train.py`)

`DOAMAMBA` (PyTorch Lightning module):
- **Backbone:** Stacked `MambaResCTF` blocks that mix along frequency, time, and channel axes independently using `CompatibleMamba` (CUDA-optimized with CPU fallback).
- **DOA head:** outputs a 2D unit vector `[B, T, 2]` (sin/cos of azimuth angle).
- **Confidence head:** outputs κ (Von Mises concentration), converted to circular std via `kappa_to_circ_std()`.

Key hyperparameters in `config.yaml`: `d_model`, `hidden_dim`, `layers` (list of expand factors), `input_dim` (= `nfft`), `freq_dim`, `time_dim`.

### Loss Functions (`confidence_localization/model.py`)

Multiple loss options exist; the active one is controlled by `bound_loss_func` in `config.yaml` (currently `"von_mises"`):
- `vm_nll_calibrated()` — Von Mises NLL with a soft coverage penalty to calibrate κ
- `von_mises_loss()` — plain Von Mises NLL
- `gaussian_loss()`, `hetero_gaussian_nll_err()`, `bound_loss()`, etc. — alternatives

Total loss = DOA loss + `log_var_weight` × confidence loss + `temporal_reg_factor` × temporal smoothness regularizer.

### Training Details

- PyTorch Lightning with 16-bit mixed precision
- AdamW optimizer with `ReduceLROnPlateau` scheduler
- Three ModelCheckpoint callbacks: best val loss, best accuracy, best calibration
- Resume from checkpoint: set `resume_from_checkpoint` in `config.yaml`
- To fine-tune only the confidence head, call `train_logstd_only()` which freezes DOA weights

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
batch_size: 16
lr: 1e-4
epochs: 100
log_var_weight: 1        # confidence loss weight
temporal_reg_factor: 1e-4

# Loss
bound_loss_func: "von_mises"

# Paths (outputs)
logs: logs/
models: models/
samples: samples/
```

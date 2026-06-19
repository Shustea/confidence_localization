"""Evaluate a trained DOAMAMBA checkpoint on a cached dataset stage.

Mirrors train.DOAMAMBA.validation_step metrics (VAD-gated, frame-weighted over
the whole set) and writes per-recording + aggregate CSVs. Use it for the held-out
RealMAN TEST set (the paper's headline number) or any cached val/eval stage.

Run:
  python confidence_localization/eval_cached.py \
    --checkpoint outputs/2026-06-09/21-03-24/models/best-acc10-epoch=155-validation_accuracy_10_vad=0.78.ckpt \
    --stage eval --device cuda:2 --out runs/realman_eval_v798
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
for p in (_REPO, _REPO / "confidence_localization", _REPO / "data"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import confidence_localization_dataloader as cld  # noqa: E402
from models import build_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=str(_REPO / "config.yaml"),
                    help="Config for architecture (hidden_dim etc.). Must match the checkpoint.")
    ap.add_argument("--stage", default="eval", choices=("train", "val", "eval"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="runs/realman_eval")
    args = ap.parse_args()

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)
    # batch_size=1: cached features have variable T; avoid pad-frame contamination.
    OmegaConf.update(cfg, "loader_per_source.realman.batch_size", 1, force_add=True)
    OmegaConf.update(cfg, "loader_per_source.realman.num_workers", 4, force_add=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model(cfg)
    state = torch.load(args.checkpoint, map_location="cpu")
    sd = state.get("state_dict", state)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[eval] loaded {args.checkpoint}")
    print(f"[eval]   missing={len(missing)} unexpected={len(unexpected)} keys (log_std head rename expected)")
    model.eval().to(device)

    loader = cld.get_dataloader(cfg, stage=args.stage)
    print(f"[eval] stage={args.stage}  recordings={len(loader.dataset)}  device={device}")

    thr10 = float(np.deg2rad(10.0)); thr15 = float(np.deg2rad(15.0))
    # Frame-weighted accumulators (VAD-gated).
    acc = dict(n_vad=0.0, sum_mae=0.0, sum_sq=0.0, n10=0.0, n15=0.0,
               n_inl=0.0, sum_inl=0.0, n_cov=0.0, n_all=0.0)
    rows = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            spectrum, labels, title, vad, _wav = batch
            spectrum = spectrum.to(device)
            doa_unit, log_std = model(spectrum)
            doa = torch.atan2(doa_unit[..., 1], doa_unit[..., 0]).cpu()
            log_std = log_std.squeeze(-1).cpu()

            safe = torch.where(torch.isfinite(labels), labels, torch.zeros_like(labels))
            err = model.circ_error(doa - safe)              # [B,T] rad in [0,pi]
            err_deg = torch.rad2deg(err)
            mask_all = torch.isfinite(labels).float()
            mask_vad = mask_all * (vad.float() if vad is not None else mask_all)
            bound = log_std.exp()

            nv = float(mask_vad.sum())
            if nv <= 0:
                continue
            in10 = ((err < thr10).float() * mask_vad).sum()
            in15 = ((err < thr15).float() * mask_vad).sum()
            inl = mask_vad * (err < thr15).float()
            cov = ((err < bound).float() * mask_vad).sum()

            acc["n_vad"] += nv
            acc["n_all"] += float(mask_all.sum())
            acc["sum_mae"] += float((err_deg * mask_vad).sum())
            acc["sum_sq"] += float(((err_deg ** 2) * mask_vad).sum())
            acc["n10"] += float(in10); acc["n15"] += float(in15)
            acc["n_inl"] += float(inl.sum()); acc["sum_inl"] += float((err_deg * inl).sum())
            acc["n_cov"] += float(cov)

            # per-recording row
            rmae = float((err_deg * mask_vad).sum() / nv)
            r10 = float(in10 / nv); r15 = float(in15 / nv)
            rows.append((str(title[0]) if isinstance(title, (list, tuple)) else str(title),
                         int(mask_vad.sum()), rmae, r10, r15))
            if (i + 1) % 250 == 0:
                print(f"  {i+1}/{len(loader.dataset)}  running acc@10={acc['n10']/max(acc['n_vad'],1):.3f}")

    nv = max(acc["n_vad"], 1.0)
    agg = {
        "n_recordings": len(rows),
        "n_vad_frames": int(acc["n_vad"]),
        "mae_vad_deg": acc["sum_mae"] / nv,
        "rmse_vad_deg": (acc["sum_sq"] / nv) ** 0.5,
        "mae_inlier15_deg": acc["sum_inl"] / max(acc["n_inl"], 1.0),
        "acc_at_10_vad": acc["n10"] / nv,
        "acc_at_15_vad": acc["n15"] / nv,
        "sigma_coverage_vad": acc["n_cov"] / nv,
        "vad_active_frac": acc["n_vad"] / max(acc["n_all"], 1.0),
    }

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with (out / "per_recording.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["title", "n_vad", "mae_vad_deg", "acc10_vad", "acc15_vad"])
        for r in rows:
            w.writerow([r[0], r[1], f"{r[2]:.4f}", f"{r[3]:.4f}", f"{r[4]:.4f}"])
    with (out / "aggregate.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(list(agg.keys())); w.writerow([f"{v:.4f}" if isinstance(v, float) else v for v in agg.values()])

    print("\n=== RealMAN", args.stage.upper(), "(VAD-gated, frame-weighted) ===")
    print(f"  recordings        : {agg['n_recordings']}")
    print(f"  VAD frames        : {agg['n_vad_frames']}")
    print(f"  MAE               : {agg['mae_vad_deg']:.2f}°")
    print(f"  MAE (inliers<15°) : {agg['mae_inlier15_deg']:.2f}°")
    print(f"  RMSE              : {agg['rmse_vad_deg']:.2f}°")
    print(f"  Acc@10°           : {100*agg['acc_at_10_vad']:.1f}%")
    print(f"  Acc@15°           : {100*agg['acc_at_15_vad']:.1f}%")
    print(f"  sigma-coverage    : {100*agg['sigma_coverage_vad']:.1f}%")
    print(f"  -> {out}/aggregate.csv")


if __name__ == "__main__":
    main()

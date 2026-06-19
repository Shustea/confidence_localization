"""Build a cropped GCC-PHAT .pt cache for the gcc_* models.

The GCC twin of ``build_realman_cache_splits.py``: same disjoint
train / val / eval split logic, but each sample stores a framed, lag-cropped
GCC-PHAT tensor (``util.gcc_phat_frames`` -> ``[P, T, n_lags]``) instead of the
RTF. The feature is written under the same ``rtf`` payload key, so the regular
``CachedExternalDataset`` loads it unchanged (point ``data.<stage>.gcc.cache_root``
at this cache and set ``data.<stage>.source: gcc``).

Splits (default mode, no train.rar):
  VAL partition  -> 80% train + 20% val   (deterministic shuffle, seed=42)
  TEST partition -> eval                   (held out)

Real-train mode (--use-real-train, requires extracted train scenes):
  TRAIN partition -> train  (scene-filtered with --scenes)
  VAL  partition  -> val
  TEST partition  -> eval

Run:
    python data/build_gcc_cache_splits.py --max-per-split 100   # quick test
    python data/build_gcc_cache_splits.py                       # full default split
    python data/build_gcc_cache_splits.py --use-real-train --max-train 5000 --splits train
"""

from __future__ import annotations

import argparse
import shutil
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (_REPO_ROOT, _REPO_ROOT / "data", _REPO_ROOT / "confidence_localization"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from eval_utils import (  # noqa: E402
    build_realman_labels,
    cache_file_path,
    load_realman_metadata,
    load_realman_waveform,
    save_cached_sample,
)
from confidence_localization_dataloader import _vad_from_waveform  # noqa: E402
from util import gcc_phat_frames  # noqa: E402

_SEED = 42


def _safe_wipe_dir(target_dir: Path) -> None:
    target_dir = Path(target_dir)
    for p in target_dir.iterdir():
        if p.name.startswith(".nfs"):
            continue
        try:
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink()
        except OSError:
            pass


def _cache_one(args):
    (cfg, root, row, channels, use_noisy, target_dir, index, overwrite, title_prefix) = args
    try:
        wav, sample_rate, base_title = load_realman_waveform(
            root, row, channels, use_noisy=use_noisy,
        )
        wav = wav.float()
        target_fs = int(cfg.fs)
        if int(sample_rate) != target_fs:
            import torchaudio.functional as taF
            wav = taF.resample(wav, int(sample_rate), target_fs)
            sample_rate = target_fs

        gcc = gcc_phat_frames(wav, cfg)                      # [P, T, n_lags] (raw)
        labels = build_realman_labels(row, gcc.shape[1])
        vad = _vad_from_waveform(cfg, wav, int(sample_rate), gcc.shape[1])
        title = f"{title_prefix}_{base_title}"
        out_path = cache_file_path(str(target_dir), index, title)
        saved = save_cached_sample(
            out_path, gcc, labels, title,
            meta={"source": "realman_gcc", "split_prefix": title_prefix,
                  "channels": list(channels), "feature": "gcc",
                  "n_lags": int(gcc.shape[-1])},
            vad=vad, overwrite=overwrite,
        )
        return saved, title, None
    except Exception as exc:
        return False, "<error>", f"{exc.__class__.__name__}: {exc}\n{traceback.format_exc()}"


def _build_split(cfg, root, split_name, channels, use_noisy, rows, target_dir,
                 workers, overwrite, title_prefix):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (cfg, root, rows[i], channels, use_noisy, str(target_dir), i, overwrite, title_prefix)
        for i in range(len(rows))
    ]
    print(f"[{split_name}] caching {len(jobs)} samples ({workers} workers) -> {target_dir}")
    n_saved = n_err = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_cache_one, j) for j in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=split_name):
            saved, title, err = fut.result()
            if err:
                n_err += 1
                print(f"  {title}: {err.splitlines()[0]}")
            elif saved:
                n_saved += 1
    print(f"[{split_name}] {n_saved} saved, {n_err} errored "
          f"({len(jobs) - n_saved - n_err} skipped/exists)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(_REPO_ROOT / "config.yaml"))
    p.add_argument("--mode", default="moving", choices=("moving", "static"))
    p.add_argument("--channels", default=None,
                   help="Comma-separated channels. Default cfg.data.train.gcc.channels.")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--use-real-train", action="store_true",
                   help="Read the train partition CSV (requires extracted train scenes).")
    p.add_argument("--scenes", default=None,
                   help="(real-train only) Comma-separated scene names to keep.")
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-val", type=int, default=None)
    p.add_argument("--max-eval", type=int, default=None)
    p.add_argument("--max-per-split", type=int, default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--no-wipe", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--splits", default="train,val,eval")
    args = p.parse_args()

    cfg = OmegaConf.load(args.config)
    realman_root = str(cfg.realman_root)
    gcc_target = Path(str(cfg.realman_gcc_target))

    if args.channels:
        channels = [int(c) for c in args.channels.split(",") if c.strip()]
    else:
        try:
            channels = list(cfg.data.train.gcc.channels)
        except Exception:
            channels = [0, 3, 5, 7, 1]
    use_noisy = True

    requested = {s.strip() for s in args.splits.split(",") if s.strip()}
    rng = np.random.default_rng(_SEED)
    train_rows, valh_rows, test_rows = [], [], []
    train_prefix = "TRAIN_M" if args.use_real_train else "VAL_M"

    if args.use_real_train:
        # Only the train partition is required on disk. train+val are a
        # deterministic val_fraction split of it (matches the 90/10 RTF cache);
        # eval still comes from the held-out test partition when present.
        if {"train", "val"} & requested:
            train_pool = load_realman_metadata(realman_root, "train", args.mode).to_dict("records")
            print(f"  TRAIN partition: {len(train_pool)} rows")
            if args.scenes:
                names = [s.strip() for s in args.scenes.split(",") if s.strip()]
                train_pool = [r for r in train_pool
                              if any(f"/{s}/" in str(r.get("filename", "")) for s in names)]
                print(f"  scene filter {names}: {len(train_pool)} rows kept")
            perm = rng.permutation(len(train_pool))
            n_val = int(round(args.val_fraction * len(train_pool))) if "val" in requested else 0
            train_rows = [train_pool[i] for i in perm[n_val:]] if "train" in requested else []
            valh_rows = [train_pool[i] for i in perm[:n_val]]
        if "eval" in requested:
            test_rows = load_realman_metadata(realman_root, "test", args.mode).to_dict("records")
    else:
        if {"train", "val"} & requested:
            val_pool = load_realman_metadata(realman_root, "val", args.mode).to_dict("records")
            print(f"  VAL partition: {len(val_pool)} rows")
            perm = rng.permutation(len(val_pool))
            n_val = int(round(args.val_fraction * len(val_pool)))
            train_rows = [val_pool[i] for i in perm[n_val:]]
            valh_rows = [val_pool[i] for i in perm[:n_val]]
        if "eval" in requested:
            test_rows = load_realman_metadata(realman_root, "test", args.mode).to_dict("records")
            print(f"  TEST partition: {len(test_rows)} rows")

    if args.max_train is not None:
        train_rows = train_rows[: int(args.max_train)]
    if args.max_val is not None:
        valh_rows = valh_rows[: int(args.max_val)]
    if args.max_eval is not None:
        test_rows = test_rows[: int(args.max_eval)]
    if args.max_per_split is not None:
        cap = int(args.max_per_split)
        train_rows, valh_rows, test_rows = train_rows[:cap], valh_rows[:cap], test_rows[:cap]

    print(f"final split sizes:  train={len(train_rows)}  val={len(valh_rows)}  eval={len(test_rows)}")

    splits = [
        ("train", train_prefix, gcc_target / "train", train_rows),
        ("val",   "VAL_M",      gcc_target / "val",   valh_rows),
        ("eval",  "TEST_M",     gcc_target / "eval",  test_rows),
    ]
    splits = [s for s in splits if s[0] in requested]
    if not splits:
        print(f"--splits={args.splits} matched nothing; valid: train, val, eval")
        return

    if not args.no_wipe:
        for split_name, _prefix, target_dir, _rows in splits:
            if target_dir.exists():
                print(f"[{split_name}] wiping existing cache dir: {target_dir}")
                _safe_wipe_dir(target_dir)

    for split_name, prefix, target_dir, rows in splits:
        if not rows:
            print(f"[{split_name}] no rows, skipping.")
            continue
        _build_split(cfg, realman_root, split_name, channels, use_noisy, rows,
                     target_dir, args.workers, args.overwrite, prefix)

    total = 0
    for split_name, _prefix, target_dir, _rows in splits:
        if not target_dir.exists():
            continue
        n_files = sum(1 for _ in target_dir.glob("*.pt"))
        size_bytes = sum(p.stat().st_size for p in target_dir.glob("*.pt"))
        total += size_bytes
        print(f"[{split_name}] final: {n_files} files, {size_bytes / 1e6:.1f} MB -> {target_dir}")
    print(f"total cache size: {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

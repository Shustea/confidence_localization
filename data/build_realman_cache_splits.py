"""Rebuild a clean RealMAN .pt cache with disjoint train / val / eval splits.

Source mappings (single-source moving mode only, matching cfg.data.train.realman.mode):

  Default mode (no train.rar downloaded):
    VAL  partition → 80% train + 20% val   (deterministic shuffle, seed=42)
    TEST partition → eval                  (held out)

  Real-train mode (--use-real-train, requires partial train.rar on disk):
    TRAIN partition → train                (RealMAN's actual train recordings)
    VAL  partition → val                   (held out)
    TEST partition → eval                  (held out)

That way no recording appears in more than one split, and the held-out eval
matches RealMAN's intended test-partition semantics.

Outputs under ``cfg.realman_target``:
  train/ <- ${realman_target}/train/000000__{VAL,TRAIN}_M_*.pt ...
  val/   <- ${realman_target}/val/000000__VAL_M_*.pt ...
  eval/  <- ${realman_target}/eval/000000__TEST_M_*.pt ...

Run:
    python data/build_realman_cache_splits.py
    # use the actual RealMAN train partition (download train scenes first):
    python data/build_realman_cache_splits.py --use-real-train --max-train 5000
    # tighter cap (e.g. test that everything works first):
    python data/build_realman_cache_splits.py --max-per-split 100
    # don't wipe existing dirs:
    python data/build_realman_cache_splits.py --no-wipe
"""

from __future__ import annotations

import argparse
import os
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
    normalize_rtf,
    preprocess_waveform,
    save_cached_sample,
)
from confidence_localization_dataloader import _vad_from_waveform  # noqa: E402

_SEED = 42


def _safe_wipe_dir(target_dir: Path) -> None:
    """Delete cache files in ``target_dir`` tolerant of NFS busy/.nfs* artifacts.

    A plain shutil.rmtree dies with [Errno 16] Device or resource busy when an
    orphaned reader (e.g. a stale train.py dataloader) still holds a deleted
    .pt open and NFS has silly-renamed it to .nfsXXXX. Those files are harmless
    to the rebuilt cache (the loader globs *.pt), so skip what we can't remove.
    """
    target_dir = Path(target_dir)
    skipped = 0
    for p in target_dir.iterdir():
        if p.name.startswith(".nfs"):
            skipped += 1
            continue
        try:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink()
        except OSError:
            skipped += 1
    if skipped:
        print(f"  ({skipped} busy/.nfs file(s) left in place — harmless)")


def _cache_one(args):
    """Worker: cache a single (row, target_dir, index) into a .pt file.

    Returns (saved: bool, title: str, err: str | None). Errors are swallowed so a
    single bad recording doesn't kill the run.
    """
    cfg, root, row, channels, use_noisy, target_dir, index, overwrite, title_prefix = args
    try:
        wav, sample_rate, base_title = load_realman_waveform(
            root, row, channels, use_noisy=use_noisy,
        )
        wav = wav.float()
        # RealMAN is 48 kHz; resample to cfg.fs so the REIR lag scale matches
        # the synthetic / LOCATA pipelines (all features share one sample rate).
        target_fs = int(cfg.fs)
        if int(sample_rate) != target_fs:
            import torchaudio.functional as taF
            wav = taF.resample(wav, int(sample_rate), target_fs)
            sample_rate = target_fs
        rtf = preprocess_waveform(
            cfg, wav, sample_rate=int(sample_rate), normalize=False,
        )
        rtf = normalize_rtf(rtf)
        labels = build_realman_labels(row, rtf.shape[1])
        vad = _vad_from_waveform(cfg, wav, int(sample_rate), rtf.shape[1])
        title = f"{title_prefix}_{base_title}"
        out_path = cache_file_path(str(target_dir), index, title)
        saved = save_cached_sample(
            out_path, rtf, labels, title,
            meta={"source": "realman", "split_prefix": title_prefix,
                  "channels": list(channels)},
            vad=vad, overwrite=overwrite,
        )
        return saved, title, None
    except Exception as exc:
        return False, "<error>", f"{exc.__class__.__name__}: {exc}\n{traceback.format_exc()}"


def _build_split(cfg, root, split_name, mode, channels, use_noisy,
                 rows, target_dir, workers, overwrite, title_prefix):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (cfg, root, rows[i], channels, use_noisy, str(target_dir), i, overwrite, title_prefix)
        for i in range(len(rows))
    ]
    print(f"[{split_name}] caching {len(jobs)} samples ({workers} workers) "
          f"-> {target_dir}")
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
    p.add_argument("--mode", default="moving",
                   choices=("moving", "static"),
                   help="RealMAN source mode CSV to read.")
    p.add_argument("--channels", default=None,
                   help="Comma-separated channel indices. Defaults to "
                        "cfg.data.train.realman.channels.")
    p.add_argument("--val-fraction", type=float, default=0.2,
                   help="(default mode only) Fraction of VAL partition that goes "
                        "to the val split (rest is train). Ignored with --use-real-train.")
    p.add_argument("--use-real-train", action="store_true",
                   help="Read train_<mode>_source_location.csv (requires a partial "
                        "train.rar to have been downloaded + extracted). Skips the "
                        "VAL 80/20 split and uses all VAL rows for val.")
    p.add_argument("--scenes", default=None,
                   help="(real-train only) Comma-separated scene names; keep only train "
                        "rows whose filename contains one of them. Use this to match the "
                        "subset of scene .rar archives actually downloaded.")
    p.add_argument("--max-train", type=int, default=None,
                   help="Cap the train split to N samples. Useful with --use-real-train "
                        "since the train CSV has tens of thousands of rows.")
    p.add_argument("--max-val", type=int, default=None,
                   help="Cap the val split to N samples (e.g. set to max_train/9 for a "
                        "90/10 train/val ratio).")
    p.add_argument("--max-eval", type=int, default=None,
                   help="Cap the eval split to N samples (default: full TEST partition).")
    p.add_argument("--max-per-split", type=int, default=None,
                   help="Cap every split to N samples (testing).")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--no-wipe", action="store_true",
                   help="Don't delete existing cache dirs before writing.")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite individual cache files if they already exist.")
    p.add_argument("--splits", default="train,val,eval",
                   help="Comma-separated subset of splits to (re)build. Other splits "
                        "are untouched. E.g. --splits train to rebuild just train.")
    p.add_argument("--rtf-noise-mode", default="energy",
                   choices=("energy", "prefix"),
                   help="Noise-covariance estimation for the RTF front-end. Real "
                        "RealMAN recordings have no clean pre-speech segment, so "
                        "'energy' (lowest-energy frames) is the correct default.")
    args = p.parse_args()

    cfg = OmegaConf.load(args.config)
    # Real recordings have no guaranteed noise prefix -> use energy-gated noise.
    cfg.rtf_noise_mode = args.rtf_noise_mode
    print(f"[front-end] rtf_noise_mode = {cfg.rtf_noise_mode}")
    realman_root = str(cfg.realman_root)
    realman_target = Path(str(cfg.realman_target))

    if args.channels:
        channels = [int(c) for c in args.channels.split(",") if c.strip()]
    else:
        try:
            channels = list(cfg.data.train.realman.channels)
        except Exception:
            channels = [0, 3, 5, 7, 1]
    use_noisy = True

    # Always need VAL and TEST metadata.
    print(f"loading RealMAN metadata: split=val mode={args.mode}")
    val_frame = load_realman_metadata(realman_root, "val", args.mode)
    val_rows = val_frame.to_dict("records")
    print(f"  VAL  partition: {len(val_rows)} rows")

    print(f"loading RealMAN metadata: split=test mode={args.mode}")
    test_frame = load_realman_metadata(realman_root, "test", args.mode)
    test_rows = test_frame.to_dict("records")
    print(f"  TEST partition: {len(test_rows)} rows")

    rng = np.random.default_rng(_SEED)
    if args.use_real_train:
        # Read the actual train CSV; train_rows come from train partition,
        # val_rows used in full (no 80/20 split).
        print(f"loading RealMAN metadata: split=train mode={args.mode}")
        train_frame = load_realman_metadata(realman_root, "train", args.mode)
        train_pool = train_frame.to_dict("records")
        print(f"  TRAIN partition: {len(train_pool)} rows")
        # Keep only rows for the scene archives actually downloaded.
        if args.scenes:
            scene_names = [s.strip() for s in args.scenes.split(",") if s.strip()]
            def _row_scene_ok(row):
                fn = str(row.get("filename", ""))
                return any(f"/{s}/" in fn or f"/{s}." in fn for s in scene_names)
            before = len(train_pool)
            train_pool = [r for r in train_pool if _row_scene_ok(r)]
            print(f"  scene filter {scene_names}: {len(train_pool)}/{before} rows kept")
        # Shuffle deterministically so a capped subset is representative.
        perm = rng.permutation(len(train_pool))
        train_rows = [train_pool[i] for i in perm]
        if args.max_train is not None:
            train_rows = train_rows[: int(args.max_train)]
        valh_rows = val_rows                              # full VAL → val
        train_prefix = "TRAIN_M"
    else:
        # Default: deterministic 80/20 split of VAL into train/val.
        perm = rng.permutation(len(val_rows))
        n_val = int(round(args.val_fraction * len(val_rows)))
        train_idx = perm[n_val:]
        val_idx = perm[:n_val]
        train_rows = [val_rows[i] for i in train_idx]
        valh_rows = [val_rows[i] for i in val_idx]
        train_prefix = "VAL_M"

    # Per-split caps (applied before the global --max-per-split cap).
    if args.max_train is not None and not args.use_real_train:
        # In default mode --max-train still trims the VAL-derived train rows.
        train_rows = train_rows[: int(args.max_train)]
    if args.max_val is not None:
        valh_rows = valh_rows[: int(args.max_val)]
    if args.max_eval is not None:
        test_rows = test_rows[: int(args.max_eval)]

    if args.max_per_split is not None:
        cap = int(args.max_per_split)
        train_rows = train_rows[:cap]
        valh_rows = valh_rows[:cap]
        test_rows = test_rows[:cap]

    print(f"final split sizes:  train={len(train_rows)}  "
          f"val={len(valh_rows)}  eval={len(test_rows)}  "
          f"(train_prefix={train_prefix})")

    requested = {s.strip() for s in args.splits.split(",") if s.strip()}
    splits = [
        ("train", train_prefix, realman_target / "train", train_rows),
        ("val",   "VAL_M",      realman_target / "val",   valh_rows),
        ("eval",  "TEST_M",     realman_target / "eval",  test_rows),
    ]
    # Only touch splits the user asked for; leave the others alone on disk.
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
        _build_split(
            cfg=cfg,
            root=realman_root,
            split_name=split_name,
            mode=args.mode,
            channels=channels,
            use_noisy=use_noisy,
            rows=rows,
            target_dir=target_dir,
            workers=args.workers,
            overwrite=args.overwrite,
            title_prefix=prefix,
        )

    # Final summary.
    total_size = 0
    for split_name, _prefix, target_dir, _rows in splits:
        if not target_dir.exists():
            continue
        n_files = sum(1 for _ in target_dir.glob("*.pt"))
        size_bytes = sum(p.stat().st_size for p in target_dir.glob("*.pt"))
        total_size += size_bytes
        print(f"[{split_name}] final: {n_files} files, {size_bytes / 1e9:.1f} GB "
              f"-> {target_dir}")
    print(f"total cache size: {total_size / 1e9:.1f} GB")


if __name__ == "__main__":
    main()

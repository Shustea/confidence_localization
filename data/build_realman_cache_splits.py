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


def scan_noise_pool(root: str) -> list[str]:
    """Return base paths (sans _CH{n}.flac) of every extracted ma_noise recording."""
    base = Path(root) / "train" / "ma_noise"
    bases: list[str] = []
    if base.exists():
        for ch0 in sorted(base.rglob("*_CH0.flac")):
            bases.append(str(ch0)[: -len("_CH0.flac")])
    return bases


def _load_noise(base: str, channels, target_fs: int):
    """Load a multichannel noise recording [C, T] (cfg-order channels), resampled."""
    import soundfile as sf
    waves, sr = [], None
    for ch in channels:
        audio, fs = sf.read(f"{base}_CH{ch}.flac", dtype="float32", always_2d=True)
        waves.append(torch.from_numpy(audio[:, 0]))
        sr = fs if sr is None else sr
    n = min(int(w.numel()) for w in waves)
    noise = torch.stack([w[:n] for w in waves], dim=0)
    if int(sr) != int(target_fs):
        import torchaudio.functional as taF
        noise = taF.resample(noise, int(sr), int(target_fs))
    return noise


def _mix_at_snr(speech, noise, snr_db: float, rng):
    """Mix speech [C,T] + noise [C,*] at speech-to-noise ratio snr_db (broadband).

    Noise is tiled if shorter than the speech, random-cropped if longer.
    """
    C, T = speech.shape
    nt = int(noise.shape[1])
    if nt < T:
        reps = (T + nt - 1) // nt
        noise = noise.repeat(1, reps)[:, :T]
    elif nt > T:
        start = int(rng.integers(0, nt - T + 1))
        noise = noise[:, start:start + T]
    sp = float((speech ** 2).mean()) + 1e-12
    npow = float((noise ** 2).mean()) + 1e-12
    scale = (sp / (npow * (10.0 ** (snr_db / 10.0)))) ** 0.5
    return speech + scale * noise


def _cache_one(args):
    """Worker: cache one (row, snr) sample into a .pt file.

    If ``noise_pool`` and ``snr`` are given, a random noise recording is mixed
    into the speech at the target SNR before the RTF is computed (waveform-level
    augmentation; the source DOA labels are unchanged). Errors are swallowed so a
    single bad recording doesn't kill the run.
    """
    (cfg, root, row, channels, use_noisy, target_dir, index, overwrite,
     title_prefix, noise_pool, snr) = args
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

        title_suffix = ""
        if noise_pool and snr is not None:
            # Seed per (recording, snr) for reproducible noise selection.
            rng = np.random.default_rng(abs(hash((base_title, float(snr)))) % (2**32))
            nb = noise_pool[int(rng.integers(0, len(noise_pool)))]
            noise = _load_noise(nb, channels, target_fs)
            wav = _mix_at_snr(wav, noise, float(snr), rng)
            title_suffix = f"__snr{int(round(float(snr))):+d}"

        rtf = preprocess_waveform(
            cfg, wav, sample_rate=int(sample_rate), normalize=False,
        )
        rtf = normalize_rtf(rtf)
        labels = build_realman_labels(row, rtf.shape[1])
        vad = _vad_from_waveform(cfg, wav, int(sample_rate), rtf.shape[1])
        title = f"{title_prefix}_{base_title}{title_suffix}"
        out_path = cache_file_path(str(target_dir), index, title)
        saved = save_cached_sample(
            out_path, rtf, labels, title,
            meta={"source": "realman", "split_prefix": title_prefix,
                  "channels": list(channels),
                  "snr_db": (float(snr) if snr is not None else None)},
            vad=vad, overwrite=overwrite,
        )
        return saved, title, None
    except Exception as exc:
        return False, "<error>", f"{exc.__class__.__name__}: {exc}\n{traceback.format_exc()}"


def _build_split(cfg, root, split_name, mode, channels, use_noisy,
                 rows, target_dir, workers, overwrite, title_prefix,
                 noise_pool=None, snr_levels=None):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    # Noise augmentation: emit one job per (recording, snr level). Without it,
    # one clean job per recording (snr=None).
    levels = snr_levels if (noise_pool and snr_levels) else [None]
    jobs, idx = [], 0
    for i in range(len(rows)):
        for snr in levels:
            jobs.append((cfg, root, rows[i], channels, use_noisy, str(target_dir),
                         idx, overwrite, title_prefix, noise_pool, snr))
            idx += 1
    aug = f" x{len(levels)} snr levels {levels}" if levels != [None] else ""
    print(f"[{split_name}] caching {len(jobs)} samples ({len(rows)} recs{aug}, "
          f"{workers} workers) -> {target_dir}")
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
    p.add_argument("--snr-levels", default=None,
                   help="Comma-separated dB SNR levels for noise augmentation of the "
                        "TRAIN split, e.g. '15,5,-5' -> 3 augmented copies per recording, "
                        "each mixing a random ma_noise recording at that speech-to-noise "
                        "ratio. Requires extracted train/ma_noise/. val/eval are never "
                        "augmented (they use real ma_noisy_speech).")
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

    # Only load the metadata CSVs for the splits actually requested, so a
    # train-only rebuild works even after the raw val/test dirs (and their CSVs)
    # have been deleted to free disk.
    requested = {s.strip() for s in args.splits.split(",") if s.strip()}
    rng = np.random.default_rng(_SEED)
    train_rows, valh_rows, test_rows = [], [], []
    train_prefix = "TRAIN_M" if args.use_real_train else "VAL_M"

    if args.use_real_train:
        if "train" in requested:
            print(f"loading RealMAN metadata: split=train mode={args.mode}")
            train_pool = load_realman_metadata(realman_root, "train", args.mode).to_dict("records")
            print(f"  TRAIN partition: {len(train_pool)} rows")
            if args.scenes:
                scene_names = [s.strip() for s in args.scenes.split(",") if s.strip()]
                def _row_scene_ok(row):
                    fn = str(row.get("filename", ""))
                    return any(f"/{s}/" in fn or f"/{s}." in fn for s in scene_names)
                before = len(train_pool)
                train_pool = [r for r in train_pool if _row_scene_ok(r)]
                print(f"  scene filter {scene_names}: {len(train_pool)}/{before} rows kept")
            perm = rng.permutation(len(train_pool))
            train_rows = [train_pool[i] for i in perm]
            if args.max_train is not None:
                train_rows = train_rows[: int(args.max_train)]
        if "val" in requested:
            print(f"loading RealMAN metadata: split=val mode={args.mode}")
            valh_rows = load_realman_metadata(realman_root, "val", args.mode).to_dict("records")
            print(f"  VAL partition: {len(valh_rows)} rows")
        if "eval" in requested:
            print(f"loading RealMAN metadata: split=test mode={args.mode}")
            test_rows = load_realman_metadata(realman_root, "test", args.mode).to_dict("records")
            print(f"  TEST partition: {len(test_rows)} rows")
    else:
        # Default: deterministic 80/20 split of the VAL partition into train/val.
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

    # Noise augmentation (TRAIN split only): mix real ma_noise at each SNR level.
    snr_levels = None
    noise_pool = []
    if args.snr_levels:
        snr_levels = [float(x) for x in args.snr_levels.split(",") if x.strip()]
        noise_pool = scan_noise_pool(realman_root)
        if not noise_pool:
            print("[noise-aug] WARNING: --snr-levels set but no extracted "
                  "train/ma_noise/ recordings found; train will be cached CLEAN.")
            snr_levels = None
        else:
            print(f"[noise-aug] {len(noise_pool)} noise recordings, "
                  f"snr_levels={snr_levels} -> train x{len(snr_levels)}")

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
        # Augment the train split only; val/eval use real ma_noisy_speech.
        split_noise_pool = noise_pool if split_name == "train" else None
        split_snr = snr_levels if split_name == "train" else None
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
            noise_pool=split_noise_pool,
            snr_levels=split_snr,
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

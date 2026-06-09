"""Replace samples whose source trajectory violates the post-fix far-field rule.

Identifies samples where the filename-encoded start/end DOA imply an angular
speed above the new cap (omega = v / source_min_dist = 1.5 / 1.5 = 1 rad/s,
so |end - start| over a sample_length_secs window must be <= 5 rad for 5 s).
Deletes the bad .wav + .pt pairs, regenerates fresh samples with the current
config (which now picks source_min_dist = 1.5 m), and preprocess the new wavs.

Run:
    python data/replace_unphysical_samples.py
    # or override paths / threshold from the CLI:
    python data/replace_unphysical_samples.py \\
        --dataset /workspaces/.../1spk_robust_5s_5ch \\
        --omega-max 1.0 --workers 16
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (_REPO_ROOT, _REPO_ROOT / "data", _REPO_ROOT / "confidence_localization"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from data_helpers import mix_signal               # noqa: E402
from make_dataset import preprocess_file          # noqa: E402

# Filename format from data_helpers.generate_speaker_sample:
#   "{start_doa:.2f}_{end_doa:.2f}_{height:.2f}_{file_id}-..."
# end_doa = start_doa + angular_speed * sample_length_secs (unwrapped, may exceed +/- pi).
_FNAME_RE = re.compile(r"^(?P<start>-?\d+\.\d+)_(?P<end>-?\d+\.\d+)_-?\d+\.\d+_")


def parse_doas(name: str):
    m = _FNAME_RE.match(name)
    if m is None:
        return None
    return float(m["start"]), float(m["end"])


def is_bad(name: str, max_delta_rad: float) -> bool:
    doas = parse_doas(name)
    if doas is None:
        return False
    return abs(doas[1] - doas[0]) > max_delta_rad


def _regen_worker(args):
    cfg, out_dir, max_delta_rad = args
    # Unique seed per worker so parallel calls don't collide.
    seed = int(time.time() * 1e6) % (2**32 - 1) ^ os.getpid()
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    torch.manual_seed(seed & 0xFFFFFFFF)
    try:
        # Retry up to N times if the regen sample is itself "bad" — shouldn't
        # happen under the new config but cheap insurance against config drift.
        for _ in range(5):
            mix, mix_id = mix_signal(cfg)
            doas = parse_doas(mix_id + ".wav")
            if doas is None or abs(doas[1] - doas[0]) <= max_delta_rad:
                break
        out_path = os.path.join(out_dir, f"{mix_id}.wav")
        i = 1
        while os.path.exists(out_path):
            out_path = os.path.join(out_dir, f"{mix_id}_{i}.wav")
            i += 1
        final_mix = mix.T.astype(np.float32)
        sf.write(out_path, final_mix / np.abs(final_mix).max(0).reshape(1, -1), cfg.fs)
        return out_path
    except Exception:
        traceback.print_exc()
        return None


def _preprocess_worker(args):
    cfg, wav_path = args
    pt_path = wav_path.replace(".wav", ".pt")
    if os.path.exists(pt_path):
        return None
    try:
        rtf = preprocess_file(cfg, wav_path)
        torch.save(rtf, pt_path)
        return None
    except Exception as e:
        return f"err {wav_path}: {e}"


def process_split(cfg, split_dir: Path, max_delta_rad: float, workers: int) -> None:
    wavs = [f for f in os.listdir(split_dir) if f.endswith(".wav")]
    bad = [f for f in wavs if is_bad(f, max_delta_rad)]
    print(f"[{split_dir.name}] {len(bad)} / {len(wavs)} wavs flagged "
          f"(|end - start| > {max_delta_rad} rad)")
    if not bad:
        return

    # 1) Delete bad .wav + .pt pairs.
    for f in bad:
        wav_path = split_dir / f
        pt_path = split_dir / f.replace(".wav", ".pt")
        for p in (wav_path, pt_path):
            try:
                p.unlink()
            except FileNotFoundError:
                pass

    # 2) Regenerate the same number of fresh samples via mix_signal().
    print(f"[{split_dir.name}] regenerating {len(bad)} samples ({workers} workers)")
    regen_jobs = [(cfg, str(split_dir), max_delta_rad)] * len(bad)
    new_wavs: list[str] = []
    with Pool(processes=workers) as pool:
        for path in tqdm(pool.imap_unordered(_regen_worker, regen_jobs),
                         total=len(regen_jobs), desc=f"{split_dir.name} regen"):
            if path:
                new_wavs.append(path)
    print(f"[{split_dir.name}] wrote {len(new_wavs)} / {len(bad)} replacement wavs")

    # 3) Preprocess the new wavs into .pt features.
    print(f"[{split_dir.name}] preprocessing {len(new_wavs)} new .pt features")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_preprocess_worker, (cfg, p)) for p in new_wavs]
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc=f"{split_dir.name} preproc"):
            err = fut.result()
            if err:
                print(" ", err)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config", default=str(_REPO_ROOT / "config.yaml"),
        help="Path to project config.yaml (used for mix_signal + preprocess).",
    )
    p.add_argument(
        "--dataset",
        default="/workspaces/confidence_localization/data/lab/data/processed/1spk_robust_5s_5ch",
        help="Dataset root; train/ and validation/ subdirs are scanned.",
    )
    p.add_argument(
        "--omega-max", type=float, default=1.0,
        help="Max angular speed in rad/s. |end - start| > omega_max * sample_length_secs is flagged.",
    )
    p.add_argument("--workers", type=int, default=16, help="Parallel workers.")
    p.add_argument("--splits", default="train,validation",
                   help="Comma-separated subdir names to process.")
    args = p.parse_args()

    cfg = OmegaConf.load(args.config)
    max_delta_rad = float(args.omega_max) * float(cfg.sample_length_secs)
    print(f"omega_max = {args.omega_max} rad/s, sample_length_secs = "
          f"{cfg.sample_length_secs}, |delta| threshold = {max_delta_rad} rad")

    dataset = Path(args.dataset)
    for split in args.splits.split(","):
        split_dir = dataset / split.strip()
        if not split_dir.is_dir():
            print(f"[{split.strip()}] missing dir, skipping")
            continue
        process_split(cfg, split_dir, max_delta_rad, args.workers)


if __name__ == "__main__":
    main()

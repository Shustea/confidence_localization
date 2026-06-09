import librosa
import shutil
import numpy as np
import soundfile as sf
from tqdm import tqdm
import subprocess
import torch
import random
import time

import traceback
from multiprocessing import Pool, cpu_count

import os
import hydra
import sys

import matplotlib.pyplot as plt

from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from confidence_localization.util import compute_multichannel_stft, estimate_rtf
from data_helpers import mix_signal

MAX_CORES_FOR_PREPROCESS = 8


def convert_wv12wav(args):
    # input - original path of csr-1, sampling rate
    # output - void
    # function converts between WV1 file to in our original path to a WAV file in our intended path, file SR is fs
    #
    # Basically this function is just a WV1->WAV converter

    if args.delete_all_samples_flag:
        if os.path.exists(args.wav_path):
            confirm = (
                input(
                    f"Are you sure you want to delete all WAV samples in '{args.wav_path}'? "
                    "This action cannot be undone. (y/n): "
                )
                .strip()
                .lower()
            )
            if confirm == "y":
                print(
                    f"!*!*!*! Doomsday Button Pressed - Deleting all WAV samples in: {args.wav_path} !*!*!*!*!"
                )
                shutil.rmtree(args.wav_path)

    if not os.path.exists(args.wav_path):
        os.makedirs(args.wav_path)

    file_paths = []

    with open(
        [
            args.original_path + "/" + file
            for file in os.listdir(args.original_path)
            if file.endswith(".tbl")
        ][0],
        "r",
    ) as tbl_file:
        for line in tbl_file:
            file_path = line.strip()
            if file_path.endswith(".wv1"):
                file_paths.append(file_path.split(" ")[-1])

    for sample_path in tqdm(file_paths):
        if not os.path.isdir(args.wav_path + sample_path.split("/")[-2]):
            os.mkdir(args.wav_path + sample_path.split("/")[-2])
        old_path = args.original_path + "/" + sample_path
        new_path = (
            args.wav_path
            + sample_path.split("/")[-2]
            + "/"
            + sample_path.split("/")[-1].split(".")[0]
            + ".wav"
        )
        subprocess.run(
            [os.getcwd() + "/data/sph2pipe.exe", "-f", "wav", old_path, new_path],
            check=True,
        )


def preprocess_file(cfg, sample_file):
    if sample_file.split(".")[-1] == "pt":
        signal = torch.load(sample_file)
    elif sample_file.split(".")[-1] == "wav":
        signal = torch.from_numpy(sf.read(sample_file, dtype="float32")[0]).T

    if len(signal.shape) > 2:
        return signal

    sample_name = os.path.splitext(os.path.basename(sample_file))[0]

    if np.isnan(signal).sum() > 0:
        raise ValueError(f"NaNs detected in wav : {sample_name}")

    stft = compute_multichannel_stft(signal, cfg)
    rtf = estimate_rtf(cfg, stft)
    # return torch.cat([torch.stack((rtf[i].real, rtf[i].imag), dim=0) for i in range(rtf.shape[0])], dim=0)
    return rtf


def process_and_save(cfg, path, file):
    try:
        full = os.path.join(path, file)
        if full.endswith("wav"):
            if not os.path.exists(full.replace(".wav", ".pt")):
                rtf = preprocess_file(cfg, full)
                torch.save(rtf, full.replace(".wav", ".pt"))
        return None  # success
    except Exception as e:
        return f"error in file {file}: {e}"


def is_real_wav(path):
    try:
        with open(path, "rb") as f:
            header = f.read(12)
        return header[:4] == b"RIFF" and header[8:12] == b"WAVE"
    except:
        return False


def fix_directory(root):
    for dirpath, _, files in os.walk(root):
        for f in files:
            full = os.path.join(dirpath, f)

            if f.lower().endswith(".pt") and f.split(".")[-2] == "wav":
                os.rename(full, full.replace(".wav", ""))

            elif f.lower().endswith(".wav"):

                if is_real_wav(full):
                    # it's a legit wav → leave it
                    continue

                # not a real wav → treat as a bad saved .pt
                print(f"[FIX] converting bad wav → pt: {full}")

                try:
                    obj = torch.load(full, map_location="cpu")
                except:
                    print(f"[SKIP] cannot load as pt: {full}")
                    continue

                # new name
                new_path = full + ".pt"

                # avoid overwriting if exists
                if os.path.exists(new_path):
                    new_path = full.replace(".wav", ".pt")

                torch.save(obj, new_path)
                os.remove(full)


def preprocess(cfg):
    print("--- Performing GEVD for all data in parallel ---")

    def process_path(path):
        files = os.listdir(path)
        with ProcessPoolExecutor(max_workers=MAX_CORES_FOR_PREPROCESS) as executor:
            futures = [
                executor.submit(process_and_save, cfg, path, file) for file in files
            ]
            for f in tqdm(as_completed(futures)):
                err = f.result()
                if err:
                    print(err)

    print(" --- started train ---")
    # fix_directory(cfg.train_path)
    process_path(cfg.train_path)
    print(" --- finished train -> starting validation ---")
    process_path(cfg.val_path)
    print(" --- finished validation ---")


def _count_pt_or_wav(dir_path):
    if not os.path.isdir(dir_path):
        return 0
    return sum(
        1 for f in os.listdir(dir_path) if (f.endswith(".pt") or f.endswith(".wav"))
    )


def _worker_make_one(args):
    """
    Worker function:
        args = (cfg, out_dir)
    """
    cfg, out_dir = args

    # ---- FIX: UNIQUE SEED PER WORKER - SO WE DONT CREATE DUPLICATES ----
    seed = int(time.time() * 1e6) % (2**32 - 1) ^ os.getpid()
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    torch.manual_seed(seed & 0xFFFFFFFF)
    # --------------------------------------------------------------------

    try:
        mix, mix_id = mix_signal(cfg)
        out = os.path.join(out_dir, f"{mix_id}.wav")

        # avoid overwrite
        if os.path.exists(out):
            i = 1
            while True:
                alt = os.path.join(out_dir, f"{mix_id}_{i}.pt")
                if not os.path.exists(alt):
                    out = alt
                    break
                i += 1

        final_mix = mix.T.astype(np.float32)

        sf.write(out, final_mix / np.abs(final_mix).max(0).reshape(1, -1), args[0].fs)
        return True

    except Exception as e:
        print(f"[worker] failed with: {e}")
        traceback.print_exc()
        return False


def create_data(cfg):
    """
    Parallel version of dataset generation.
    Uses multiprocessing.Pool for speed.
    """
    n_workers = getattr(cfg, "num_workers", cpu_count())

    for name, target, out_dir in [
        ("train", int(cfg.train_size), cfg.train_path),
        ("val", int(cfg.val_size), cfg.val_path),
    ]:
        os.makedirs(out_dir, exist_ok=True)
        existing = _count_pt_or_wav(out_dir)
        to_make = max(0, target - existing)

        print(
            f"[{name}] have {existing}/{target}, creating {to_make} using {n_workers} workers..."
        )

        if to_make == 0:
            print(f"[{name}] nothing to do.")
            continue

        # pack arguments for pool
        jobs = [(cfg, out_dir)] * to_make

        made = 0
        with Pool(processes=n_workers) as pool:
            for ok in tqdm(pool.imap_unordered(_worker_make_one, jobs), total=to_make):
                if ok:
                    made += 1

        print(f"[{name}] done: {existing + made}/{target}")


def fix_ptpt_files(directory):
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith(".pt.pt"):
            full_path = os.path.join(directory, filename)
            try:
                os.remove(full_path)
            except Exception as e:
                print(f"Failed to delete {filename}: {e}")


# --- LOCATA caching --------------------------------------------------------

LOCATA_SINGLE_SOURCE_TASKS = (1, 3, 5)


def _locata_base(locata_root, split):
    from pathlib import Path
    base = Path(locata_root)
    if (base / split).exists():
        return base / split
    if (base / "LOCATA" / split).exists():
        return base / "LOCATA" / split
    raise FileNotFoundError(f"Could not find '{split}' under '{locata_root}'.")


def _discover_locata_pairs(locata_root, split, array, tasks=LOCATA_SINGLE_SOURCE_TASKS):
    base = _locata_base(locata_root, split)
    pairs = []
    for t in tasks:
        td = base / f"task{t}"
        if not td.exists():
            continue
        for rd in sorted(td.iterdir()):
            if not (rd.is_dir() and rd.name.startswith("recording") and (rd / array).exists()):
                continue
            try:
                r = int(rd.name.replace("recording", ""))
            except ValueError:
                continue
            pairs.append((t, r))
    return pairs


def _parse_locata_pairs(value):
    """Accept ``'1:1,1:2,3:1'`` strings or ``[[1,1],[1,2],...]`` / ``[{'task':1,'recording':1},...]`` lists."""
    if value is None:
        return None
    if isinstance(value, str):
        out = []
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            t, r = item.split(":")
            out.append((int(t), int(r)))
        return out
    out = []
    for item in value:
        if isinstance(item, dict):
            out.append((int(item["task"]), int(item["recording"])))
        else:
            t, r = item
            out.append((int(t), int(r)))
    return out


def _resample_audio(wav, fs_in, fs_out):
    if int(fs_in) == int(fs_out):
        return wav
    import torchaudio.functional as taF
    return taF.resample(wav, int(fs_in), int(fs_out))


def _cache_one_locata(
    cfg, locata_root, split, task, recording, array, channels, out_dir, index, overwrite,
):
    # Local imports — these modules are only needed when caching LOCATA.
    from eval_utils import (
        build_locata_labels,
        cache_file_path,
        load_locata_waveform,
        normalize_rtf,
        preprocess_waveform,
        save_cached_sample,
    )
    from confidence_localization_dataloader import _vad_from_waveform

    wav, sample_rate, title, _ = load_locata_waveform(
        str(locata_root), split, task, recording, array, channels,
    )
    wav = wav.float()
    wav = _resample_audio(wav, int(sample_rate), int(cfg.fs))
    rtf = preprocess_waveform(cfg, wav, sample_rate=int(cfg.fs), normalize=False)
    rtf = normalize_rtf(rtf)
    labels = build_locata_labels(
        str(locata_root), split, task, recording, array, rtf.shape[1],
    )
    vad = _vad_from_waveform(cfg, wav, int(cfg.fs), rtf.shape[1])
    out_path = cache_file_path(str(out_dir), index, title)
    saved = save_cached_sample(out_path, rtf, labels, title, vad=vad, overwrite=overwrite)
    return out_path, saved, tuple(rtf.shape), int(labels.shape[0]), int(vad.sum().item())


def _stage_locata_cfg(cfg, stage):
    data = cfg.get("data") if hasattr(cfg, "get") else None
    if data is None or stage not in data:
        return None
    section = data[stage]
    return section.get("locata") if hasattr(section, "get") else getattr(section, "locata", None)


def create_locata_cache(cfg):
    """Cache LOCATA recordings into ``cfg.data.<stage>.locata.cache_root`` for each stage.

    Per-stage settings (under ``cfg.data.<stage>.locata``):
        * ``root``         — LOCATA raw root (defaults to ``cfg.locata_root``).
        * ``split``        — ``dev`` | ``eval``.
        * ``array``        — array name (default ``eigenmike``).
        * ``channels``     — 0-indexed channel list to extract.
        * ``cache_root``   — output directory for the cache.
        * ``pairs``        — optional list of ``[task, recording]`` (or
          ``"task:rec,..."``). Falls back to every single-source eigenmike
          recording under the split.
        * ``overwrite``    — bool, default False.

    Pull the run via the make_dataset CLI:
        python data/make_dataset.py +cache_locata=true
        # optional CLI overrides:
        python data/make_dataset.py +cache_locata=true \\
            +data.train.locata.pairs='1:1,1:2,1:3,3:1' \\
            +data.val.locata.pairs='3:2,3:3,5:1,5:2,5:3'
    """
    stages = []
    for stage in ("train", "val", "eval", "test"):
        locata_cfg = _stage_locata_cfg(cfg, stage)
        if locata_cfg is None:
            continue
        if locata_cfg.get("cache_root", None) is None:
            print(f"[locata-cache] {stage}: no cache_root set — skipping.")
            continue
        stages.append((stage, locata_cfg))

    if not stages:
        print("[locata-cache] no LOCATA stages configured under cfg.data.*.locata — nothing to do.")
        return

    for stage, locata_cfg in stages:
        root = locata_cfg.get("root", None) or cfg.locata_root
        split = str(locata_cfg.get("split", "dev"))
        array = str(locata_cfg.get("array", "eigenmike"))
        channels = list(locata_cfg.get("channels", [0, 1]))
        cache_root = locata_cfg.get("cache_root")
        overwrite = bool(locata_cfg.get("overwrite", False))

        pairs = _parse_locata_pairs(locata_cfg.get("pairs", None))
        if pairs is None:
            pairs = _discover_locata_pairs(root, split, array)

        if not pairs:
            print(f"[locata-cache] {stage}: no recordings found in {root}/{split} for {array}.")
            continue

        os.makedirs(cache_root, exist_ok=True)
        print(f"[locata-cache] {stage}: caching {len(pairs)} recording(s) -> {cache_root}")
        n_written = 0
        for idx, (t, r) in enumerate(pairs):
            try:
                path, saved, rtf_shape, n_labels, n_vad = _cache_one_locata(
                    cfg, root, split, t, r, array, channels, cache_root, idx, overwrite,
                )
            except Exception as exc:
                print(f"  task{t} rec{r}: SKIP ({exc})")
                continue
            status = "wrote" if saved else "skip (exists)"
            n_written += int(saved)
            print(f"  task{t} rec{r}: {status} {path.name}  rtf={rtf_shape}  "
                  f"labels={n_labels}  vad_active={n_vad}")
        print(f"[locata-cache] {stage}: done ({n_written}/{len(pairs)} written).")


@hydra.main(config_path="..", config_name="config", version_base="1.1")
def main(cfg, convert_wv12wav_flag=False, create_data_flag=False, preprocess_flag=True):
    if bool(cfg.get("cache_locata", False)):
        create_locata_cache(cfg)
        return
    # cfg overrides (Hydra CLI: +create_data=true) win over Python kwargs.
    do_convert = bool(cfg.get("convert_wv12wav", convert_wv12wav_flag))
    do_create = bool(cfg.get("create_data", create_data_flag))
    do_preprocess = bool(cfg.get("preprocess", preprocess_flag))
    if do_convert:
        convert_wv12wav(cfg)
    if do_create:
        create_data(cfg)
    if do_preprocess:
        preprocess(cfg)


if __name__ == "__main__":
    main()

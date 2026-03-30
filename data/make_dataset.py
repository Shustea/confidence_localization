import librosa
import shutil
import numpy as np
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
from pathlib import Path
import torch, soundfile as sf


from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_helpers import mix_signal
from eval_utils import (
    build_locata_labels,
    build_realman_labels,
    cache_file_path,
    limit_records,
    load_locata_waveform,
    load_realman_metadata,
    load_realman_waveform,
    preprocess_waveform,
    save_cached_sample,
)

MAX_CORES_FOR_PREPROCESS = 8


def _cfg_get(node, key, default=None):
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    if hasattr(node, "get"):
        value = node.get(key, default)
        return default if value is None else value
    return getattr(node, key, default)


def convert_wv12wav(args):
    # input - original path of csr-1, sampling rate
    # output - void
    # function converts between WV1 file to in our original path to a WAV file in our intended path, file SR is fs
    #
    # Basically this function is just a WV1->WAV converter
    

    """Convert WSJ0 `.wv1` files into `.wav` files under the configured raw-data directory."""
    if args.delete_all_samples_flag:
        if os.path.exists(args.wav_path):
            confirm = input(
                f"Are you sure you want to delete all WAV samples in '{args.wav_path}'? "
                "This action cannot be undone. (y/n): "
            ).strip().lower()
            if confirm == 'y':
                print(f"!*!*!*! Doomsday Button Pressed - Deleting all WAV samples in: {args.wav_path} !*!*!*!*!")
                shutil.rmtree(args.wav_path)

    if not os.path.exists(args.wav_path):
        os.makedirs(args.wav_path)

    file_paths = []

    with open([args.original_path + '/' + file for file in os.listdir(args.original_path) if file.endswith('.tbl')][0], 'r') as tbl_file:
        for line in tbl_file:
                file_path = line.strip()
                if file_path.endswith('.wv1'):
                    file_paths.append(file_path.split(' ')[-1])

    for sample_path in tqdm(file_paths):
        if not os.path.isdir(args.wav_path + sample_path.split('/')[-2]): 
            os.mkdir(args.wav_path + sample_path.split('/')[-2])
        old_path = args.original_path + '/' + sample_path
        new_path = args.wav_path + sample_path.split('/')[-2] + '/' + sample_path.split('/')[-1].split('.')[0] + '.wav'
        subprocess.run([os.getcwd() + '/data/sph2pipe.exe', '-f', 'wav', old_path, new_path], check=True)


def preprocess_file(cfg, sample_file):
        """Load one waveform or cached tensor and convert waveforms into RTF features when needed.

        Example:
            Input: a ``.wav`` file holding multichannel audio or an existing ``.pt`` tensor.
            Output: either the cached tensor directly or a newly computed RTF tensor
            with shape compatible with the training dataloader.
        """
        if sample_file.split('.')[-1] == 'pt':
            signal = torch.load(sample_file)
        elif sample_file.split('.')[-1] == 'wav':
            signal = torch.from_numpy(sf.read(sample_file, dtype='float32')[0]).T

        if len(signal.shape) > 2:
            return signal
        
        sample_name = os.path.splitext(os.path.basename(sample_file))[0]

        if np.isnan(signal).sum() > 0:
            raise ValueError(f"NaNs detected in wav : {sample_name}")

        return preprocess_wave(cfg, signal)


def process_and_save(cfg, path, file):
    """Preprocess a single waveform file and save the resulting `.pt` features next to it."""
    try:
        full = os.path.join(path, file)
        if full.endswith('wav'):
            if not os.path.exists(full.replace(".wav", ".pt")):
                rtf = preprocess_file(cfg, full)
                torch.save(rtf, full.replace(".wav", ".pt"))
        return None  # success
    except Exception as e:
        return f"error in file {file}: {e}"


def is_real_wav(path):
    """Check whether a file has a valid RIFF/WAVE header."""
    try:
        with open(path, "rb") as f:
            header = f.read(12)
        return header[:4] == b"RIFF" and header[8:12] == b"WAVE"
    except:
        return False


def fix_directory(root):
    """Repair mislabeled cached feature files inside a directory tree."""
    for dirpath, _, files in os.walk(root):
        for f in files:
            full = os.path.join(dirpath, f)

            if f.lower().endswith(".pt") and f.split('.')[-2] == 'wav':
                os.rename(full, full.replace('.wav', ''))

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
    """Precompute RTF features for the configured train and validation directories in parallel."""
    print('--- Performing GEVD for all data in parallel ---')

    def process_path(path):
        """Handle path."""
        files = os.listdir(path)
        with ProcessPoolExecutor(max_workers=MAX_CORES_FOR_PREPROCESS) as executor:
            futures = [executor.submit(process_and_save, cfg, path, file) for file in files]
            for f in tqdm(as_completed(futures)):
                err = f.result()
                if err:
                    print(err)

    print(' --- started train ---')
    # fix_directory(cfg.train_path)
    process_path(cfg.train_path)
    print(' --- finished train -> starting validation ---')
    process_path(cfg.val_path)
    print(' --- finished validation ---')


def _count_pt_or_wav(dir_path):
    """Count cached feature or waveform files in a directory."""
    if not os.path.isdir(dir_path):
        return 0
    return sum(1 for f in os.listdir(dir_path) if (f.endswith(".pt") or f.endswith(".wav")))


def _worker_make_one(args):
    """
    Worker function:
        args = (cfg, out_dir)
    """
    cfg, out_dir = args

    # ---- FIX: UNIQUE SEED PER WORKER - SO WE DONT CREATE DUPLICATES ----
    seed = int(time.time() * 1e6) % (2**32 - 1) ^ os.getpid()
    random.seed(seed)
    np.random.seed(seed & 0xffffffff)
    torch.manual_seed(seed & 0xffffffff)
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
        
        sf.write(out, final_mix / np.abs(final_mix).max(0).reshape(1,-1), args[0].fs)
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
        ("val",   int(cfg.val_size),   cfg.val_path),
    ]:
        os.makedirs(out_dir, exist_ok=True)
        existing = _count_pt_or_wav(out_dir)
        to_make  = max(0, target - existing)

        print(f"[{name}] have {existing}/{target}, creating {to_make} using {n_workers} workers...")

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
    """Delete accidental `.pt.pt` duplicates from a directory."""
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith('.pt.pt'):
            full_path = os.path.join(directory, filename)
            try:
                os.remove(full_path)
            except Exception as e:
                print(f"Failed to delete {filename}: {e}")

                
def preprocess_wave(cfg, wav, sample_rate=None):
    """Convert a multichannel waveform tensor into the raw cached RTF representation."""
    if torch.isnan(wav).any():
        raise ValueError("NaNs in wav")
    return preprocess_waveform(cfg, wav, sample_rate=sample_rate, normalize=False)


def save_pt(out_path, tensor):
    """Save a tensor to disk, creating parent directories as needed."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(tensor.cpu(), out_path)


def _external_stage_cfg(cfg, stage):
    data_cfg = _cfg_get(cfg, "data")
    return _cfg_get(data_cfg, stage) if data_cfg is not None else None


def realman_make_cache(cfg, dataset_cfg, overwrite=False):
    """Precompute and cache RealMAN samples as raw RTF tensors plus labels/title sidecars."""
    root = _cfg_get(dataset_cfg, "root", _cfg_get(cfg, "realman_root"))
    split = _cfg_get(dataset_cfg, "split", "val")
    mode = _cfg_get(dataset_cfg, "mode", "moving")
    use_noisy = bool(_cfg_get(dataset_cfg, "use_noisy", True))
    channels = list(_cfg_get(dataset_cfg, "channels", _cfg_get(dataset_cfg, "chs", [0, 1])))
    every_nth = _cfg_get(dataset_cfg, "every_nth", 1)
    max_items = _cfg_get(dataset_cfg, "max_items", None)
    cache_root = _cfg_get(dataset_cfg, "cache_root", _cfg_get(cfg, "realman_target"))

    frame = load_realman_metadata(root, split, mode)
    rows = limit_records(frame.to_dict("records"), every_nth=every_nth, max_items=max_items)
    os.makedirs(cache_root, exist_ok=True)

    for idx, row in enumerate(tqdm(rows, desc=f"realman:{split}:{mode}")):
        title = Path(str(row["filename"])).stem
        out_path = cache_file_path(cache_root, idx, title)
        if out_path.exists() and not overwrite:
            continue

        wav, sample_rate, title = load_realman_waveform(root, row, channels, use_noisy=use_noisy)
        feat = preprocess_wave(cfg, wav, sample_rate=sample_rate)
        labels = build_realman_labels(row, feat.shape[1])
        save_cached_sample(
            out_path,
            feat,
            labels,
            title,
            meta={"source": "realman", "split": split, "mode": mode, "use_noisy": use_noisy, "channels": channels},
            overwrite=overwrite,
        )


def locata_make_cache(cfg, dataset_cfg, overwrite=False):
    """Precompute and cache one LOCATA recording as a raw RTF tensor plus labels/title sidecars."""
    root = _cfg_get(dataset_cfg, "root", _cfg_get(cfg, "locata_root"))
    split = _cfg_get(dataset_cfg, "split", "dev")
    task = int(_cfg_get(dataset_cfg, "task", 1))
    recording = int(_cfg_get(dataset_cfg, "recording", 1))
    array = _cfg_get(dataset_cfg, "array", "eigenmike")
    channels = list(_cfg_get(dataset_cfg, "channels", _cfg_get(dataset_cfg, "chs", [0, 1])))
    source_name = _cfg_get(dataset_cfg, "source_name", None)
    cache_root = _cfg_get(dataset_cfg, "cache_root", _cfg_get(cfg, "locata_target"))

    os.makedirs(cache_root, exist_ok=True)
    title = f"locata_task{task}_rec{recording}_{array}"
    out_path = cache_file_path(cache_root, 0, title)
    if out_path.exists() and not overwrite:
        return

    wav, sample_rate, title, _ = load_locata_waveform(root, split, task, recording, array, channels)
    feat = preprocess_wave(cfg, wav, sample_rate=sample_rate)
    labels = build_locata_labels(root, split, task, recording, array, feat.shape[1], source_name=source_name)
    save_cached_sample(
        out_path,
        feat,
        labels,
        title,
        meta={"source": "locata", "split": split, "task": task, "recording": recording, "array": array, "channels": channels},
        overwrite=overwrite,
    )


def preprocess_external(cfg):
    """Precompute cached external-dataset features for the configured stages."""
    precompute_cfg = _cfg_get(cfg, "precompute", {})
    stages = _cfg_get(precompute_cfg, "external_stages", ["eval"])
    seen = set()

    for stage in stages:
        stage_cfg = _external_stage_cfg(cfg, str(stage))
        if stage_cfg is None:
            print(f"[precompute] missing stage config: {stage}")
            continue

        source = str(_cfg_get(stage_cfg, "source", "synthetic")).lower()
        if source not in {"realman", "locata"}:
            print(f"[precompute] skipping stage '{stage}' because source='{source}'")
            continue

        dataset_cfg = _cfg_get(stage_cfg, source, stage_cfg)
        cache_root = _cfg_get(dataset_cfg, "cache_root", None)
        key = (source, str(cache_root))
        if key in seen:
            continue
        seen.add(key)

        overwrite = bool(_cfg_get(dataset_cfg, "overwrite_cache", _cfg_get(precompute_cfg, "overwrite", False)))
        if source == "realman":
            realman_make_cache(cfg, dataset_cfg, overwrite=overwrite)
        else:
            locata_make_cache(cfg, dataset_cfg, overwrite=overwrite)


def preprocess_test(cfg):
    """Backward-compatible alias for external feature precomputation."""
    preprocess_external(cfg)


@hydra.main(config_path="..", config_name="config", version_base="1.1")
def main(cfg, convert_wv12wav_flag=False, create_data_flag=False, preprocess_flag=False, preprocess_test_flag=False, preprocess_external_flag=False):
    """Serve as the Hydra entrypoint for waveform conversion, synthetic-data generation, and preprocessing tasks."""
    if convert_wv12wav_flag:
        convert_wv12wav(cfg)
    if create_data_flag:
        create_data(cfg)
    if preprocess_flag:
        preprocess(cfg)
    if preprocess_test_flag or preprocess_external_flag:
        preprocess_external(cfg)



if __name__ == "__main__":
    main()

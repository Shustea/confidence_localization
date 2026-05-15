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


@hydra.main(config_path="..", config_name="config", version_base="1.1")
def main(cfg, convert_wv12wav_flag=False, create_data_flag=False, preprocess_flag=True):
    if convert_wv12wav_flag:
        convert_wv12wav(cfg)
    if create_data_flag:
        create_data(cfg)
    if preprocess_flag:
        preprocess(cfg)


if __name__ == "__main__":
    main()

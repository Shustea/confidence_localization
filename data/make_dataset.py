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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from confidence_localization.util import compute_multichannel_stft, estimate_rtf
from data_helpers import compute_nb_img, mix_signal

MAX_CORES_FOR_PREPROCESS = 8

def convert_wv12wav(args):
    # input - original path of csr-1, sampling rate
    # output - void
    # function converts between WV1 file to in our original path to a WAV file in our intended path, file SR is fs
    #
    # Basically this function is just a WV1->WAV converter
    

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


def sanity_check_gevd(args, theta=0):
    fs   = float(args.fs)
    c    = float(args.sound_velocity)
    f0   = 1000.0
    T    = 1.0
    hop  = int(args.win_len * (1 - args.overlap))
    lam  = c / f0

    # geometry: two mics on x-axis at 0 and 0.5*lambda
    azim_gt = torch.deg2rad(torch.tensor(float(theta), dtype=torch.float64))
    mic_pos = lam * torch.tensor([[0.0, 0.0, 0.0],
                                  [0.5, 0.0, 0.0]], dtype=torch.float64)
    d = float(torch.abs(mic_pos[1, 0] - mic_pos[0, 0]))  # spacing

    # synth data
    tau = (mic_pos[:, 0] * torch.cos(azim_gt)) / c  # per-mic delay
    t = torch.arange(int(fs*T), dtype=torch.float64) / fs
    x = torch.sin(2*torch.pi*f0*(t[None, :] - tau[:, None]))  # (M,N)
    x = x + 1e-6*torch.randn_like(x)

    # STFT
    spect = torch.stft(x,
                       n_fft=args.win_len,
                       hop_length=hop,
                       center=False,
                       return_complex=True)  # (M,F,T)

    # RTF estimate: disable VAD (all zeros)
    Fbins, Ts = spect.shape[1], spect.shape[-1]
    vad = torch.zeros(Fbins, Ts, dtype=torch.bool, device=spect.device)
    rtf = estimate_rtf(spect, vad_mask=vad)  # expect shape (M-1,F,T) w.r.t. ref mic 0

    # pick tone bin by power in mic 0
    power_f = (spect[0].abs()**2).mean(dim=-1)         # (F,)
    tone_bin = int(torch.argmax(power_f).item())

    # take the single RTF channel between mic1 and mic0, average over time
    # If your estimate_rtf returns (M,F,T) instead of (M-1,F,T), change to:
    # rtf_bin = spect[1, tone_bin, :] / (spect[0, tone_bin, :] + 1e-12)
    rtf_bin = rtf[0, tone_bin, :]                      # (T,)
    rtf_mean = torch.mean(rtf_bin)                     # complex scalar

    # unwrap to principal value [-pi, pi]
    phi = torch.atan2(rtf_mean.imag, rtf_mean.real).item()

    # effective frequency of chosen bin
    freqs = torch.fft.rfftfreq(args.win_len, d=1.0/fs).numpy()
    f_eff = float(freqs[tone_bin])
    k = 2.0 * np.pi * f_eff / c

    # model: phi ≈ -k d cos(theta)
    cos_theta = -phi / (k * d)

    # clamp numerical noise and compute angle
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta_est = float(np.arccos(cos_theta))            # radians, in [0, pi]

    return np.rad2deg(theta_est)

def sweep_and_plot(cfg, save_path="/workspaces/confidence_localization/samples/"):
    # sweep GT azimuths in radians
    thetas = np.linspace(0, 180, 46)   # 1° steps
    theta_est = []

    for th in tqdm(thetas):
        # call your sanity check with configurable ground-truth azimuth
        est = sanity_check_gevd(cfg, theta=th) 
        theta_est.append(est)

    theta_est_deg = np.array(theta_est)

    plt.figure(figsize=(8,6))
    plt.plot(thetas, theta_est_deg,
             linewidth=2.5, color="navy", label="Estimated DOA")
    plt.plot([0,180], [0,180], "k--", linewidth=1.5, label="Ideal (y=x)")

    plt.xlabel("Ground-truth azimuth [deg]", fontsize=14)
    plt.ylabel("Estimated DOA [deg]", fontsize=14)
    plt.title("GEVD Sanity Check Sweep", fontsize=16, fontweight="bold")
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    # save instead of showing
    plt.savefig(save_path + '/gevd_sanity_sweep.png', dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8,6))
    plt.plot(thetas[:-1], theta_est_deg[:-1]-thetas[:-1],
             linewidth=2.5, color="navy", label="Estimated DOA error")

    plt.xlabel("Ground-truth azimuth [deg]", fontsize=14)
    plt.ylabel("Estimated DOA error[deg]", fontsize=14)
    plt.title("GEVD Sanity Check Sweep Error", fontsize=16, fontweight="bold")
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    # save instead of showing
    plt.savefig(save_path + '/gevd_sanity_sweep_error.png', dpi=300, bbox_inches="tight")
    plt.close()

def circ_error_deg(est_deg, gt_deg):
    """
    Compute circular error between estimated DOA and ground truth DOA in degrees,
    handling ambiguity modulo 180°.

    Parameters
    ----------
    est_deg : float or array-like
        Estimated DOA(s) in degrees.
    gt_deg : float or array-like
        Ground truth DOA(s) in degrees.

    Returns
    -------
    error : float or np.ndarray
        Circular error(s) in degrees, in the range [-90, 90].
    """
    est = np.asarray(est_deg)
    gt  = np.asarray(gt_deg)

    # Raw difference
    diff = est - gt

    # Wrap to [-180, 180)
    diff = (diff + 180) % 360 - 180

    # Because of phase-DOA ambiguity, wrap further to [-90, 90]
    diff = (diff + 90) % 180 - 90

    return diff

def create_rir_bank(cfg):
    torch.cuda.set_device(7)
    rir_bank_path = cfg.rir_bank_path

    room_dim = cfg.room_dim
    T60 = cfg.T60
    fs = cfg.fs
    Tmax = 1.5 * T60
    c = cfg.sound_velocity

    beta = rir.beta_SabineEstimation(room_dim, T60)
    azimuths = np.linspace(0, 2 * np.pi, 360)
    heights = np.arange(-0.1, 0.1, 0.025) + cfg.src_height
    distances = np.arange(0.9, 1.1, 0.1)

    array_center = cfg.receivers_coords[0]
    array_center_x, array_center_y = array_center[:2]

    mic_positions = np.array(cfg.receivers_coords)
    num_mics = mic_positions.shape[0]
    nb_img = rir.t2n(Tmax, room_dim, c)

    for height_idx, h in enumerate(heights):
        src_positions = []
        for theta in azimuths:
            for r in distances:
                src_x = array_center_x + r * np.cos(theta)
                src_y = array_center_y + r * np.sin(theta)
                src_positions.append([src_x, src_y, h])

        src_positions = np.array(src_positions)
        total_sources = len(src_positions)
        batch_size = 8
        all_rirs = []

        for i in tqdm(range(0, total_sources, batch_size), desc=f"Height {height_idx+1}/{len(heights)}"):
            batch_src = src_positions[i:i + batch_size]
            batch_src_exp = np.repeat(batch_src, num_mics, axis=0)
            batch_rcv_exp = np.tile(mic_positions, (len(batch_src), 1))

            rirs = rir.simulateRIR(
                room_sz=room_dim,
                beta=beta,
                pos_src=batch_src_exp,
                pos_rcv=batch_rcv_exp,
                nb_img=nb_img,
                Tmax=Tmax,
                fs=fs,
                c=c,
                mic_pattern='omni'
            )
            all_rirs.append(rirs)

            del rirs
            torch.cuda.empty_cache()

        rir_chunk = np.concatenate(all_rirs, axis=0)

        height_folder = os.path.join(rir_bank_path, f"height_z{int(h*1e3)}")
        os.makedirs(height_folder, exist_ok=True)

        save_path = os.path.join(
            height_folder,
            f"rir_bank_T60-{T60*100}_x{room_dim[0]}_y{room_dim[1]}_z{int(h*1e3)}.pt"
        )

        torch.save({
            'rir_chunk': rir_chunk,
            'height': h,
            'height_idx': height_idx,
            'azimuths': azimuths,
            'distances': distances,
            'mic_positions': mic_positions,
            'Tmax': Tmax,
            'fs': fs,
            'T60': T60,
            'room_dim': room_dim
        }, save_path, pickle_protocol=5)
        torch.cuda.empty_cache()

def preprocess_file(cfg, sample_file):
        signal = torch.load(sample_file)
        if signal.shape == (16, 257, 626):
            return signal
        sample_name = os.path.splitext(os.path.basename(sample_file))[0]

        if np.isnan(signal).sum() > 0:
            raise ValueError(f"NaNs detected in wav : {sample_name}")

        stft = compute_multichannel_stft(signal, cfg)
        rtf = estimate_rtf(stft)
        return torch.cat([torch.stack((rtf[i].real, rtf[i].imag), dim=0) for i in range(rtf.shape[0])], dim=0)

def process_and_save(cfg, path, file):
    try:
        rtf = preprocess_file(cfg, os.path.join(path, file))
        torch.save(rtf, os.path.join(path, file))
        return None  # success
    except Exception as e:
        return f"error in file {file}: {e}"

def preprocess(cfg):
    print('--- Performing GEVD for all data in parallel ---')

    def process_path(path):
        files = os.listdir(path)
        with ProcessPoolExecutor(max_workers=MAX_CORES_FOR_PREPROCESS) as executor:
            futures = [executor.submit(process_and_save, cfg, path, file) for file in files]
            for f in tqdm(as_completed(futures)):
                err = f.result()
                if err:
                    print(err)

    print(' --- started train ---')
    process_path(cfg.train_path)
    print(' --- finished train -> starting validation ---')
    process_path(cfg.val_path)
    print(' --- finished validation ---')

def _count_pt(dir_path):
    if not os.path.isdir(dir_path):
        return 0
    return sum(1 for f in os.listdir(dir_path) if f.endswith(".pt"))


def _worker_make_one(args):
    """
    Worker function:
        args = (cfg, out_dir)
    """
    cfg, out_dir = args

    # ---- FIX: UNIQUE SEED PER WORKER ----
    seed = int(time.time() * 1e6) % (2**32 - 1) ^ os.getpid()
    random.seed(seed)
    np.random.seed(seed & 0xffffffff)
    torch.manual_seed(seed & 0xffffffff)
    # --------------------------------------

    try:
        mix, mix_id = mix_signal(cfg)
        out = os.path.join(out_dir, f"{mix_id}.pt")

        # avoid overwrite
        if os.path.exists(out):
            i = 1
            while True:
                alt = os.path.join(out_dir, f"{mix_id}_{i}.pt")
                if not os.path.exists(alt):
                    out = alt
                    break
                i += 1

        torch.save(mix, out)
        return True

    except Exception as e:
        print(f"[worker] failed with: {e}")
        traceback.print_exc()
        return False


def _count_pt(path):
    return sum(f.endswith(".pt") for f in os.listdir(path))


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
        existing = _count_pt(out_dir)
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
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith('.pt.pt'):
            full_path = os.path.join(directory, filename)
            try:
                os.remove(full_path)
            except Exception as e:
                print(f"Failed to delete {filename}: {e}")


@hydra.main(config_path="..", config_name="config", version_base="1.1")
def main(cfg, sanity_check_flag=False, convert_wv12wav_flag=False, create_data_flag=True, preprocess_flag=False):

    # import soundfile as sf
    # print('start')
    # for dirpath,_,files in os.walk(cfg.wav_path):
    #     for f in files:
    #         if f.endswith(".wav"):
    #             try: sf.info(os.path.join(dirpath,f))
    #             except Exception as e: print("Corrupt WAV:", f, e)
    # print('end')

    if sanity_check_flag:
        sweep_and_plot(cfg, save_path="/workspaces/confidence_localization/samples")
    if convert_wv12wav_flag:
        convert_wv12wav(cfg)
    if create_data_flag:
        create_data(cfg)
    if preprocess_flag:
        preprocess(cfg)


if __name__ == "__main__":
    main()

        
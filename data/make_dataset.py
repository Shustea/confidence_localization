import librosa
import shutil
import numpy as np
import soundfile as sf
from tqdm import tqdm
import subprocess
import torch

import os
import hydra
import sys

from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gpuRIR as rir

from confidence_localization.util import compute_multichannel_stft, estimate_rtf
from data_helpers import compute_nb_img, mix_signal

MAX_CORES_FOR_PREPROCESS = 8

def convert_wv12wav(args):
    # input - original path of csr-1, sampling rate
    # output - void
    # function converts between WV1 file to in our original path to a WAV file in our intended path, file SR is fs 

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


def sanity_check_gevd(args):
    """
    One clean tone → STFT → estimate_rtf() → DOA
    Succeeds if |error| ≲ 1° for a well-behaved RTF estimator.
    """
    # ------------------------------------------------------------------ #
    # 0.  Convenience aliases & numeric hygiene
    # ------------------------------------------------------------------ #
    fs   = args.fs
    c    = args.sound_velocity          # m s-1
    f0   = 1000.0                       # test tone [Hz]
    T    = 1.0                          # signal duration [s]
    hop  = int(args.win_len * (1 - args.overlap))

    # ------------------------------------------------------------------ #
    # 1.  Microphone geometry & ground-truth source
    # ------------------------------------------------------------------ #
    azim_gt = torch.deg2rad(torch.tensor(124.5))
    mic_pos = torch.as_tensor(args.receivers_coords)  # (M,3)
    centre  = mic_pos.mean(0)
    src_pos = centre + torch.tensor([2.0*torch.cos(azim_gt),
                                     2.0*torch.sin(azim_gt),
                                     0.0])

    # geometric (far-field) delays, referenced to mic 0
    dists  = torch.linalg.norm(mic_pos - src_pos, dim=1)          # metres
    tau    = (dists - dists[0]) / c                               # seconds

    # ------------------------------------------------------------------ #
    # 2.  Synthesise the multichannel tone
    # ------------------------------------------------------------------ #
    t = torch.arange(int(fs*T)) / fs               # (Nt,)
    x = torch.sin(2*torch.pi*f0*(t[None, :] - tau[:, None]))      # (M,Nt)
    x += 1e-6*torch.randn_like(x)                                 # tiny noise

    # ------------------------------------------------------------------ #
    # 3.  STFT  →  relative transfer function (RTF)
    # ------------------------------------------------------------------ #
    spect = torch.stft(x,
                       n_fft=args.win_len,
                       hop_length=hop,
                       return_complex=True)                       # (M,F,Ts)
    rtf = estimate_rtf(spect)                                     # (M-1,F,Ts)

    # ------------------------------------------------------------------ #
    # 4.  Pick the bin that actually contains the test tone
    # ------------------------------------------------------------------ #
    power_f = (spect[0].abs()**2).mean(dim=-1)                    # (F,)
    tone_bin = torch.argmax(power_f).item()

    rtf_bin  = rtf[:, tone_bin, :]                                # (M-1,Ts)
    rtf_mean = rtf_bin.mean(dim=-1)                               # (M-1,)

    # ------------------------------------------------------------------ #
    # 5.  Phase → differential time-of-arrival (TDOA)
    # ------------------------------------------------------------------ #
    phi      = torch.angle(rtf_mean)                                # radians
    tau_hat = phi / (2*torch.pi*f0)                                 # seconds (M-1,)

    # prepend zero for the reference mic so vectors align with mic_pos
    tau_hat = torch.cat([torch.zeros(1),
                         tau_hat])                                # (M,)

    # ------------------------------------------------------------------ #
    # 6.  Least-squares DOA for arbitrary 2-D arrays
    #       (A u = c·tau   with u = [cosθ, sinθ]^⊤,  ||u||=1)
    # ------------------------------------------------------------------ #
    A  = mic_pos[:, :2] - mic_pos[0, :2]                          # (M,2)
    b  = (c * tau_hat).unsqueeze(-1)                              # (M,1)

    # Solve A_rel u = b_rel (ignore the first row which is all zeros)
    sol = torch.linalg.lstsq(A[1:], b[1:]).solution.squeeze()     # (2,)
    u   = sol / torch.norm(sol)                                   # unit vector
    theta_est = torch.atan2(u[1], u[0])                           # rad

    # ------------------------------------------------------------------ #
    # 7.  Pretty print
    # ------------------------------------------------------------------ #
    def deg(x): return float(torch.rad2deg(x))
    err = deg(abs(theta_est - torch.remainder(azim_gt, 2*torch.pi)) % (2 * np.pi)) 
    print(f"Ground-truth azimuth                : {deg(torch.remainder(azim_gt, 2*torch.pi)):6.2f}°")
    print(f"Estimated DOA (phase LS)           : {deg(theta_est):6.2f}°")
    print(f"Absolute error                     : {err:6.2f}°")
    print('---finished sanity checks!---')



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

def create_data(cfg):
    for _ in tqdm(range(cfg.train_size)):
        try:
            mix, mix_id = mix_signal(cfg)
            torch.save(mix, f"{cfg.train_path}/{mix_id}.pt")
        except:
            print("---failed---")
    print('---finished train!---')
    for _ in tqdm(range(cfg.val_size)):
        try:
            mix, mix_id = mix_signal(cfg)
            torch.save(mix, f"{cfg.val_path}/{mix_id}.pt")
        except:
            print("---failed---")
    print('---finished validation!---')

def fix_ptpt_files(directory):
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith('.pt.pt'):
            full_path = os.path.join(directory, filename)
            try:
                os.remove(full_path)
            except Exception as e:
                print(f"Failed to delete {filename}: {e}")


@hydra.main(config_path="..", config_name="config", version_base="1.1")
def main(cfg, sanity_check_flag=True, convert_wv12wav_flag=False, calculate_rirs=False, create_data_flag=False, preprocess_flag=False):
    if sanity_check_flag:
        sanity_check_gevd(cfg)
    if convert_wv12wav_flag:
        convert_wv12wav(cfg)
    if calculate_rirs:
        create_rir_bank(cfg)
    if create_data_flag:
        create_data(cfg)
    if preprocess_flag:
        preprocess(cfg)


if __name__ == "__main__":
    main()

        
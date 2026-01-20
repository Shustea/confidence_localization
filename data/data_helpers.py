import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import torch
from data.signal_generator import SignalGenerator

import os

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

def compute_nb_img(room_sz, Tmax, c=340):
    max_dist = c * Tmax
    return np.ceil(max_dist / np.array(room_sz)).astype(int)

def mix_signal(args):
    snr = np.random.uniform(args.snr_db - 3, args.snr_db + 3)

    speakers_dirs = get_speaker_dirs(args.wav_path)
    if not speakers_dirs:
        raise ValueError("No speaker directories found in the specified path.")

    first_file_id = second_file_id = 'NONE'

    beta = [args.beta]      

    mixed_signal, first_file_id, sp_path = generate_speaker_sample(args, speakers_dirs, beta)
    
    mixed_signal, noise_type = add_pink_noise(mixed_signal, sp_path,args, snr_db=snr)

    mix_id = f"{first_file_id}-{second_file_id}-beta{args.beta}-{noise_type}#"
    return mixed_signal, mix_id

def get_speaker_dirs(wav_path):
    return [d for d in os.listdir(wav_path) if os.path.isdir(os.path.join(wav_path, d))]

def sample_non_overlapping_doa(first_doa, min_sep=np.deg2rad(25), max_iter=100):
    for _ in range(max_iter):
        doa = np.random.uniform(0, 2 * np.pi)
        if np.abs(np.angle(np.exp(1j*(doa - first_doa)))) > min_sep:
            return doa

def choose_start_offset(sample_length, min_overlap=0.3, max_overlap=0.7):
    # force overlap only for part of the utterance
    overlap_fraction = np.random.uniform(min_overlap, max_overlap)
    max_start = int(sample_length * (1 - overlap_fraction))
    return np.random.randint(0, max_start)

def generate_speaker_sample(args, speakers_dirs, beta):
    sample_length = int(args.sample_length_secs * args.fs)
    sample_time = args.sample_length_secs
    speaker_id = np.random.choice(speakers_dirs)
    file_id = np.random.choice(os.listdir(os.path.join(args.wav_path, speaker_id)))
    audio, _ = librosa.load(os.path.join(args.wav_path, speaker_id, file_id), sr=args.fs)

    mic_centers = np.array(args.receivers_coords).mean(0)

    doa, dist = np.random.uniform(0, 2 * np.pi), np.random.uniform(1.5, 2.15)
    start_pos = np.array([mic_centers[0] + dist * np.cos(doa), mic_centers[1] + dist * np.sin(doa), np.random.uniform(1.65, 1.85)])

    walk_mode = np.random.rand()

    if walk_mode < 0.2:
        # 20% of the time the person stands still
        v = 0.0
        angular_speed = 0.0
    else:
        v = np.random.uniform(0.8, 1.5)
        direction = np.random.choice([-1, 1])   # Clockwise or Counter Clockwise
        angular_speed = direction * (v / dist)

    sample = pad_or_trim(audio, sample_length)

    noise_length = int(args.pre_speech_noise_time * args.fs)
    sample = np.concatenate((np.zeros(noise_length), sample))

    sample, sp_path = apply_rir_on_sample(sample, args, start_pos, dist, angular_speed, sample_length, beta)
    
    id_string = f"{doa:.2f}_{(doa + (angular_speed*sample_time)):.2f}_{start_pos[2]:.2f}_{file_id.split('.')[0]}"
    return sample, id_string, sp_path

def generate_second_speaker_sample(args, speakers_dirs, first_id, first_sample, beta, doa_range):
    sample_length = int(args.sample_length_secs * args.fs)
    sample_time = args.sample_length_secs
    remaining = [s for s in speakers_dirs if s not in first_id]
    if not remaining:
        return np.zeros_like(first_sample), 'NONE'

    speaker_id = np.random.choice(remaining)
    file_id = np.random.choice(os.listdir(os.path.join(args.wav_path, speaker_id)))
    audio, _ = librosa.load(os.path.join(args.wav_path, speaker_id, file_id), sr=args.fs)

    snr = np.random.normal(0, 5)
    gain = np.sqrt(10 ** (-snr / 10) * (np.std(first_sample) / np.std(audio)) ** 2)
    audio *= gain

    doa = sample_non_overlapping_doa(doa_range[0], min_sep=np.deg2rad(args.min_sep))
    dist = np.random.uniform(1.2, 1.3)
    angular_speed = np.random.uniform(-1, 1)
    if doa_range[0] < doa < doa_range[1]:
        prob = np.random.rand()
        if prob > 0.5:
            doa = np.random.uniform(0, doa_range[0])
        else:
            doa = np.random.uniform(doa_range[1], 2 * np.pi)

    start_pos = np.array([1.5 + dist * np.cos(doa), 1.5 + dist * np.sin(doa), np.random.uniform(1.65, 1.85)])
    raw_sample = apply_rir_on_sample(audio, args, start_pos, dist, angular_speed, sample_length, beta)

    sample = np.zeros_like(first_sample)
    start = choose_start_offset(sample_length, min_overlap=args.min_overlap, max_overlap=args.max_overlap)
    length = min(sample_length - start, raw_sample.shape[0])
    sample[start:start + length] = raw_sample[:length]

    id_string = f"{gain:.3f}_{doa:.2f}_{(doa + (angular_speed*sample_time)):.2f}_{start_pos[2]:.2f}_{file_id.split('.')[0]}"
    return sample, id_string

import numpy as np

def add_diffuse_noise(signal, args, beta, print_stats=False):
    """
    Adds diffuse noise to a multichannel signal by summing many
    weak noise sources rendered with room RIRs. Includes detailed
    statistics to verify noise quality.
    """

    T, M = signal.shape
    fs = args.fs
    mic_positions = np.array(args.receivers_coords)
    num_sources = args.num_noise_sources

    # --- realistic SNR range for diffuse noise ---
    target_snr_db = np.random.uniform(35, 40) 
    signal_power = np.mean(signal**2)
    noise_power_target = signal_power / (10 ** (target_snr_db / 10))

    # --- accumulate many weak spatial noise sources ---
    diffuse_noise = np.zeros_like(signal)

    for _ in range(num_sources):

        # small-energy white noise source
        noise_src = 0.1 * np.random.randn(T)

        # random room location (uniform)
        start_pos = np.array([
            np.random.uniform(0, args.room_dim[0]),
            np.random.uniform(0, args.room_dim[1]),
            np.random.uniform(1.2, 2.6),  # avoid floor bounce bias
        ])

        # no motion (diffuse static source)
        angular_speed = 0.0
        dist = np.random.uniform(0.5, 3.0)   # random radius for RIR generator

        # RIR spatialization
        noise_mics = apply_rir_on_sample(
            noise_src, args, start_pos, dist, angular_speed, T, beta
        )  # (T, M)

        # safety: avoid accidental gigantic reflections / NaN
        if not np.isfinite(noise_mics).all():
            continue

        diffuse_noise += noise_mics

    # --- rescale to meet total noise power ---
    noise_power = np.mean(diffuse_noise**2)
    if noise_power < 1e-12:
        # fallback to independent noise if something went wrong
        diffuse_noise = np.random.randn(T, M)
        noise_power = np.mean(diffuse_noise**2)

    gain = np.sqrt(noise_power_target / (noise_power + 1e-12))
    diffuse_noise *= gain

    noisy_signal = signal + diffuse_noise

    # --- stats printing ---
    if print_stats:
        print("\n========== DIFFUSE NOISE STATS ==========")
        print(f"Requested SNR (dB):     {target_snr_db:.2f}")
        print(f"Signal power:           {signal_power:.4e}")
        print(f"Target noise power:     {noise_power_target:.4e}")
        print(f"Actual noise power:     {np.mean(diffuse_noise**2):.4e}")

        # per-mic RMS
        rms = np.sqrt(np.mean(diffuse_noise**2, axis=0))
        print("\nPer-mic RMS:")
        for i, r in enumerate(rms):
            print(f"  Mic {i}: {r:.5f}")

        # correlation structure (diffuse field should have small correlations)
        corr = np.corrcoef(diffuse_noise.T)
        print("\nInter-mic correlation matrix:")
        print(corr)

        avg_offdiag = np.mean(np.abs(corr - np.eye(M)))
        print(f"\nAvg |corr| off-diagonal: {avg_offdiag:.4f}")
        print("==========================================\n")

    noise_id = f"diffuse_{num_sources}_sources_SNR{target_snr_db:.1f}"
    return noisy_signal, noise_id

import numpy as np

def pinknoise(N, seed=None):
    rng = np.random.default_rng(seed)
    Nfft = int(2 ** np.ceil(np.log2(N)))
    X = rng.normal(size=Nfft // 2 + 1) + 1j * rng.normal(size=Nfft // 2 + 1)
    freqs = np.arange(1, Nfft // 2 + 1)
    X[1:] /= np.sqrt(freqs)
    x = np.fft.irfft(X, n=Nfft)
    x = x[:N]
    x -= x.mean()
    x /= x.std() + 1e-12
    return x

def add_pink_noise(x, sp_path, cfg, snr_db=35):
    noise = pinknoise(x.shape[1])

    noise_placement, _ = sample_noise_positions(cfg, sp_path=sp_path)
    sp_path_noise = np.ones((x.shape[1], 1)) @ noise_placement
    rp_path = np.repeat(np.array(cfg.receivers_coords).T[None,:,:], x.shape[1], axis=0)

    gen = SignalGenerator()
    out = gen.generate(
        noise.tolist(),
        cfg.sound_velocity,
        cfg.fs,
        rp_path.tolist(),
        sp_path_noise.tolist(),
        cfg.room_dim,
        [cfg.beta],
        cfg.nsample,
        "o",
        cfg.order,
        hp_filter=False,
    )

    noise_out = np.array(out.output)

    signal_power = np.mean(x**2, axis=1, keepdims=True)
    noise_power = np.mean(noise_out**2, axis=1, keepdims=True)

    target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    gain = np.sqrt(target_noise_power / noise_power)

    signal_w_noise = x + noise_out * gain

    return signal_w_noise, f'pink_noise_{snr_db}db'

def pad_or_trim(signal, target_length):
    if len(signal.shape) == 1:
        padded = np.zeros(target_length)
    else:
        padded = np.zeros((target_length, signal.shape[-1]))
    length = min(target_length, signal.shape[0])
    padded[:length] = signal[:length]
    return padded

def apply_rir_on_sample(src, args, start_pos, distance, angular_speed, length, beta):
    fs = args.fs
    mic_positions = np.array(args.receivers_coords)  # (M,3)
    room_dim = np.array(args.room_dim)
    c = args.sound_velocity
    M = mic_positions.shape[0]

    nsample = args.nsample
    order = args.order
    hop = args.hop

    # Allocate paths 
    s_path = np.zeros((length, 3))
    r_path = np.zeros((length, 3, M))

    # Receiver paths are fixed in time
    for m in range(M):
        r_path[:, :, m] = mic_positions[m]

    # Compute initial params
    array_center = mic_positions[0][:2]
    initial_angle = np.arctan2(start_pos[1] - array_center[1],
                               start_pos[0] - array_center[0])

    # Fill in motion in hop-sized blocks
    for i in range(0, length, hop):
        t = i / fs
        angle = initial_angle + angular_speed * t

        x = array_center[0] + distance * np.cos(angle)
        y = array_center[1] + distance * np.sin(angle)
        z = start_pos[2]

        s_path[i:i+hop] = np.array([x, y, z])

    # Run simulator
    gen = SignalGenerator()
    result = gen.generate(
        input_signal=list(src[:length]),
        c=c,
        fs=fs,
        r_path=r_path.tolist(),
        s_path=s_path.tolist(),
        L=room_dim.tolist(),
        beta_or_tr=beta,
        nsamples=nsample,
        mtype="o",
        order=order,
        hp_filter=False
    )

    return np.array(result.output), s_path

def generate_ar_noise(order, coef, size, loc=0.0, scale=1e-4, burn_in=None):
    rng = np.random.default_rng()
    eps = rng.normal(loc, scale, size + (burn_in or 0))
    ar = eps.copy()
    for t in range(order, len(ar)):
        ar[t] = np.dot(coef, ar[t-order:t][::-1]) + eps[t]
    return ar[-size:]

def _as_vec3(x, name):
    a = np.asarray(x, dtype=float).reshape(-1)
    if a.size != 3:
        raise ValueError(f"{name} must be length-3, got shape {np.asarray(x).shape}")
    return a

def _as_mat3(x, name):
    a = np.asarray(x, dtype=float)
    a = np.atleast_2d(a)
    if a.shape[-1] != 3:
        raise ValueError(f"{name} must be (*,3), got shape {a.shape}")
    return a.reshape(-1, 3)

def sample_noise_positions(cfg, LL=None, rp=None, sp_path=None, J=None,
                           d_rp=None, d_sp=None, margin=None, batch=None, max_iter=None):
    LL = _as_vec3(LL if LL is not None else cfg.room_dim, "LL")
    rp = _as_mat3(rp if rp is not None else cfg.receivers_coords, "rp")
    J  = int(J if J is not None else cfg.J)

    d_rp     = float(d_rp     if d_rp     is not None else getattr(cfg, "noise_d_rp", 1.5))
    d_sp     = float(d_sp     if d_sp     is not None else getattr(cfg, "noise_d_sp", 1.5))
    margin   = float(margin   if margin   is not None else getattr(cfg, "noise_margin", 0.15))
    batch    = int(batch      if batch    is not None else getattr(cfg, "noise_batch", 4096))
    max_iter = int(max_iter   if max_iter is not None else getattr(cfg, "noise_max_iter", 2000))

    if sp_path is None or len(sp_path) == 0:
        sp = np.empty((0, 3), float)
    else:
        sp_path = _as_mat3(sp_path, "sp_path")
        N = sp_path.shape[0]
        idx = np.unique(np.round(np.linspace(0, N - 1, min(N, 300))).astype(int))
        sp = sp_path[idx]

    P = np.empty((0, 3), float)
    thr_rp2, thr_sp2 = d_rp*d_rp, d_sp*d_sp

    for _ in range(max_iter):
        C = margin + (LL - 2*margin) * np.random.rand(batch, 3)

        dr2 = ((C[:, None, :] - rp[None, :, :])**2).sum(2)
        ok = dr2.min(1) >= thr_rp2

        if sp.size:
            ds2 = ((C[:, None, :] - sp[None, :, :])**2).sum(2)
            ok &= (ds2.min(1) >= thr_sp2)

        A = C[ok]
        if A.size:
            need = J - P.shape[0]
            P = np.vstack([P, A[:need]])
            if P.shape[0] >= J:
                break

    return P, (P.shape[0] == J)

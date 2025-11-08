import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import torch
import gpuRIR as rir

import os

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

def compute_nb_img(room_sz, Tmax, c=340):
    max_dist = c * Tmax
    return np.ceil(max_dist / np.array(room_sz)).astype(int)

def mix_signal(args):
    sample_length = int(args.sample_length_secs * args.fs)
    shape = (sample_length, 1, len(args.receivers_coords))
    first_sample = np.zeros(shape)
    second_sample = np.zeros(shape)

    speakers_dirs = get_speaker_dirs(args.wav_path)
    if not speakers_dirs:
        raise ValueError("No speaker directories found in the specified path.")

    prob = np.random.rand()
    first_file_id = second_file_id = 'NONE'

    beta = rir.beta_SabineEstimation(np.array(args.room_dim), args.T60)

    if prob > args.p_silence:
        first_sample, first_file_id, doa_range = generate_speaker_sample(args, speakers_dirs, beta)

    if prob > args.p_silence + args.p_one:
        second_sample, second_file_id = generate_second_speaker_sample(args, speakers_dirs, first_file_id, first_sample, beta, doa_range)

    mixed_signal = first_sample + second_sample
    for _ in range(args.num_noise_sources):
        mixed_signal, _ = add_noise(mixed_signal, args, beta)

    mix_id = f"{first_file_id}-{second_file_id}_T60-{int(1000*args.T60)}ms_{args.num_noise_sources}#"
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

    doa, dist = np.random.uniform(0, 2 * np.pi), np.random.uniform(0.8, 1)
    start_pos = np.array([1.5 + dist * np.cos(doa), 1.5 + dist * np.sin(doa), np.random.uniform(1.65, 1.85)])
    angular_speed = np.random.uniform(-1, 1)
    sample = apply_rir_on_sample(audio, args, start_pos, dist, angular_speed, sample_length, beta)
    sample = pad_or_trim(sample, sample_length)

    id_string = f"{doa:.2f}_{(doa + (angular_speed*sample_time)):.2f}_{start_pos[2]:.2f}_{file_id.split('.')[0]}"
    return sample, id_string, (doa, (doa + angular_speed * sample_time) % (2 * np.pi))

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

def add_noise(signal, args, beta):
    sample_length = int(args.sample_length_secs * args.fs)
    snr = np.random.normal(25, 5)
    gain = np.sqrt(10 ** (-snr / 10) * (np.std(signal) ** 2 / 1e-4))
    noise = generate_ar_noise(args.ar_order, args.ar_coef, sample_length)

    doa, dist = np.random.uniform(0, 2 * np.pi), np.random.normal(1, 0.17)
    start_pos = np.array([1.5 + dist * np.cos(doa), 1.5 + dist * np.sin(doa), 1.5])
    angular_speed = 0
    noise_sample = apply_rir_on_sample(noise, args, start_pos, dist, angular_speed, sample_length, beta)

    mixed = signal + gain * noise_sample[:sample_length]
    noise_id = f"{gain:.3f}_{start_pos[0]:.2f}_{start_pos[1]:.2f}_{start_pos[2]:.2f}"
    return mixed, noise_id

def pad_or_trim(signal, target_length):
    padded = np.zeros((target_length, signal.shape[1], signal.shape[2]))
    length = min(target_length, signal.shape[0])
    padded[:length] = signal[:length]
    return padded

def apply_rir_on_sample(src, args, start_pos, distance, angular_speed, length, beta):
    torch.cuda.set_device(3)
    fs = args.fs
    mic_positions = np.array(args.receivers_coords)
    room_dim = np.array(args.room_dim)
    T60 = args.T60
    c = args.sound_velocity

    nsample = length
    time = np.arange(nsample) / fs

    # Compute source trajectory (circular motion)
    array_center = mic_positions[0][:2]
    initial_angle = np.arctan2(start_pos[1] - array_center[1], start_pos[0] - array_center[0])
    angles = initial_angle + angular_speed * time

    src_positions = np.stack([
        array_center[0] + distance * np.cos(angles),
        array_center[1] + distance * np.sin(angles),
        np.full_like(angles, start_pos[2])
    ], axis=1)  # shape: (nsample, 3)

    # Compute RIRs for moving source
    Tmax = 1.5 * T60
    nb_img = np.array(rir.t2n(Tmax, room_dim))

    RIRs = simulateRIR_with_cache(src_positions, mic_positions, room_dim, beta, nb_img, Tmax, fs, c)

    # Apply RIRs to the source signal
    filtered = rir.simulateTrajectory(np.asarray(src, dtype=np.float32).flatten(), RIRs)  # shape: (nsample, n_mics)

    output = np.expand_dims(filtered, axis=1)  # shape: (nsample, 1, n_mics)
    torch.cuda.empty_cache()
    return output

def generate_ar_noise(order, coef, size, loc=0.0, scale=1e-4, burn_in=None):
    rng = np.random.default_rng()
    eps = rng.normal(loc, scale, size + (burn_in or 0))
    ar = eps.copy()
    for t in range(order, len(ar)):
        ar[t] = np.dot(coef, ar[t-order:t][::-1]) + eps[t]
    return ar[-size:]

def simulateRIR_with_cache(pos_src_list, mic_positions, room_sz, beta, nb_img, Tmax, fs, c, cache=None):
    from collections import defaultdict

    mic_center = mic_positions[0][:2]  # or mean
    cache = cache or {}

    for pos in pos_src_list:
        key = rir_key_from_pos(pos, mic_center)
        if key in cache:
            continue
        else:
            # Simulate new RIRs for all mics
            h_mics = rir.simulateRIR(
                room_sz=np.array(room_sz, dtype=np.float32),
                beta=np.array(beta, dtype=np.float32),
                pos_src=np.repeat(pos[None, :], mic_positions.shape[0], axis=0).astype(np.float32),
                pos_rcv=mic_positions.astype(np.float32),
                nb_img=np.array(nb_img, dtype=np.int32),
                Tmax=Tmax,
                fs=fs,
                c=c,
                mic_pattern="omni"
            )
            cache[key] = h_mics

    return np.concatenate(list(cache.values()), axis=0)

def rir_key_from_pos(pos, mic_center, angular_res=1.0, distance_res=0.1):
    delta = pos[:2] - mic_center
    azimuth = np.arctan2(delta[1], delta[0]) % (2 * np.pi)
    distance = np.linalg.norm(delta)

    azim_bin = int(np.round(np.rad2deg(azimuth) / angular_res))
    dist_bin = int(np.round(distance / distance_res))
    height_bin = int(np.round(pos[2] * 1000))  # in mm

    return (azim_bin, dist_bin, height_bin)
import os
import torch
from numpy import isnan, zeros, concatenate
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import one_hot
import soundfile as sf
import re

import sys
sys.path.append('./confidence_localization/util')

from util import compute_multichannel_stft, estimate_rtf

_g_factor_pattern = re.compile(r'-([-\d.]+)_[\d.-]+_[\d.-]+_')

class CLDataset(Dataset):
    def __init__(self, cfg, root_dir, transform=None):
        self.root_dir = root_dir
        self.sample_rate = cfg.fs
        self.cfg = cfg
        self.transform = transform
        self.classification = cfg.classification
        self.sample_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.pt')]

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx):
        sample_name = self.sample_files[idx]
        rtf = torch.load(sample_name)
        speakers = list(reversed(get_speakers_from_path(sample_name[:-3])))
        title = torch.tensor(list(reversed(get_speaker_doa_from_path(sample_name[:-3])[:len(speakers)])))
        
        labels = assign_gt_to_tf_bin(self.cfg, rtf[0].shape, sample_name, self.classification)
        
        # rtf = [
        #     rtf[i] + 1j * rtf[i + 1]
        #     for i in range(0, rtf.shape[0], 2)
        # ]
        # rtf = torch.stack(rtf, dim=-1)

        # rtf = rtf / (rtf.abs() + 1e-8)
        rtf = (rtf - rtf.mean(0)) / (rtf.std(0) + 1e-6)

        if self.transform:
            rtf = self.transform(rtf)

        return rtf, labels, str(title)[8:-2]

def estimate_prtf(spectrums, win_len=4):
    M, F, T = spectrums.shape
    rtf_complex = torch.zeros((M-1, F, T), dtype=torch.complex64)
    win_len = win_len or T

    for m in range(M-1):
        for f in range(F):        
            for t_idx in range(T):
                t_start = max(0 , (t_idx-win_len))
                t_end = min(T , (t_idx+win_len))
                Xf = spectrums[0,f,t_start:t_end].reshape(1,-1)
                Xs = spectrums[m+1,f,t_start:t_end].reshape(-1,1)
                rtf_complex[m, f, t_idx] = (Xf @ Xs)/(Xf @ Xf.T)

    rtf = torch.cat([torch.stack((rtf_complex[i].real, rtf_complex[i].imag), dim=0) for i in range(rtf_complex.shape[0])], dim=0)
    return rtf

def assign_gt_to_tf_bin(cfg, spectrum_shape_tuple, path, classification):
    all_spectra = []
    energy_factor = 0 if classification else 0.2

    speakers = list(reversed(get_speakers_from_path(path[:-3])))

    F, T = spectrum_shape_tuple

    labels = torch.full((cfg.max_num_of_speakers, F, T), torch.nan)

    doas = torch.tensor(list(reversed(get_speaker_doa_from_path(path[:-3])[:len(speakers)])))
    g = list(reversed(get_g_factor_from_path(path[:-3])))

    for speaker_idx, speaker in enumerate(speakers):
        file_path = os.path.join(cfg.wav_path, speaker[:3], f"{speaker}.wav")

        # Check file existence and validity
        if not os.path.exists(file_path):
            print(f"Missing file: {file_path}")
            continue

        if os.path.getsize(file_path) == 0:
            print(f"Empty file: {file_path}")
            continue

        try:
            signal, _ = sf.read(file_path)

        except Exception as e:
            print(f"Failed to read {file_path}: {e}")
            continue
        
        target_length = cfg.fs * cfg.sample_length_secs
        signal = concatenate([signal, zeros(target_length - len(signal))]) if len(signal) < target_length else signal[:target_length]

        if speaker_idx > 0 and g[speaker_idx] is not None:
            signal = float(g[speaker_idx]) * signal

        original_spectrum = torch.stft(torch.from_numpy(signal), n_fft=cfg.win_len, hop_length=int(cfg.win_len * (1 - cfg.overlap)), return_complex=True).abs()
        all_spectra.append(original_spectrum)

    for speaker_idx in range(len(speakers)):
        spectrum = all_spectra[speaker_idx]

        if energy_factor > 0:
            vad_mask = spectrum >= (energy_factor * torch.median(spectrum))
        else:
            vad_mask = torch.ones_like(spectrum, dtype=torch.bool)

        doa_start, doa_end = doas[speaker_idx]

        doa_map = torch.linspace(doa_start, doa_end, spectrum.shape[-1]).repeat(spectrum.shape[-2], 1)

        labels[speaker_idx, vad_mask] = doa_map[vad_mask]

    return torch.remainder(labels.permute(0, 2, 1), 2 * torch.pi)

def get_speaker_positions_from_path(path):
    # Split on hyphens that come after a non-digit/letter (i.e., real separator)
    segments = re.split(r'(?<=[a-zA-Z0-9])-(?=[^0-9-])|(?<=[a-zA-Z])-(?=\d)', path[:-3].split('/')[-1])
    positions = []
    for i, speaker in enumerate(segments):
        if speaker == 'NONE':
            continue
        parts = speaker.strip('#').split('_')
        coords = parts[:3] if i == 0 else parts[1:4]
        try:
            coords = [float(c) for c in coords]
            positions.append(coords)
        except ValueError:
            print(f"Invalid coordinates in: {speaker}")
    return positions

def get_g_factor_from_path(path: str):
    g_factor = [None]
    match = _g_factor_pattern.search(path)
    if match:
        g_factor.append(float(match.group(1)))
    return g_factor

def get_speakers_from_path(path):
    segments = re.split(r'(?<=[a-zA-Z0-9])-(?=[^0-9-])|(?<=[a-zA-Z])-(?=\d)', path[:-3].split('/')[-1])
    speakers = []
    for s in segments:
        if s == 'NONE':
            continue
        parts = s.strip('#').split('_')
        speakers.append(parts[-1])
    return speakers

def get_speaker_doa_from_path(path):
    return [tuple(float(s) for s in p[:2]) for p in get_speaker_positions_from_path(path)]

def get_dataloader(cfg, root_dir, shuffle=False, transform=None):
    dataset = CLDataset(cfg, root_dir, transform=transform)
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle, num_workers=cfg.num_workers, pin_memory=True)

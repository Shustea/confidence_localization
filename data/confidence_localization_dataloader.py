import os
import torch
import math
from numpy import isnan, zeros, concatenate, array, argmax
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import one_hot
import soundfile as sf
import re

import sys
sys.path.append('./confidence_localization/util')

from util import compute_multichannel_stft, estimate_rtf

_g_factor_pattern = re.compile(r'-([-\d.]+)_[\d.-]+_[\d.-]+_')
pi = math.pi

class CLDataset(Dataset):
    def __init__(self, cfg, root_dir, transform=None):
        self.root_dir = root_dir
        self.sample_rate = cfg.fs
        self.cfg = cfg
        self.transform = transform
        self.T60 = cfg.T60
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
        
        rtf = (rtf - rtf.mean(-1).unsqueeze(-1)) / (rtf.std(-1).unsqueeze(-1) + 1e-6) # Normalize over F

        if self.transform:
            rtf = self.transform(rtf)

        return rtf, labels, str(title)[8:-2]

# def normalize_data(rtf, cfg):
#     if cfg.data_standardization == "minmax":
#         return rtf / (rtf.max() - rtf.min())
 
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
    energy_factor = 0 if classification else 0

    speakers = list(reversed(get_speakers_from_path(path[:-3])))

    F, T = spectrum_shape_tuple

    labels = torch.full((cfg.max_num_of_speakers, T), torch.nan)

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

        signal_power = torch.tensor(signal**2)
        all_spectra.append(signal_power.median())

    for speaker_idx in range(len(speakers)):
        doa_start, doa_end = doas[speaker_idx]

        doa_map = torch.linspace(doa_start, doa_end, labels.shape[1]) # Since we assume uniform speed in the simulation

        labels[speaker_idx] = doa_map

    return torch.remainder(labels[argmax(array(all_spectra))] + pi, 2 * pi) - pi

def get_speaker_positions_from_path(path):
    # Split on hyphens that come after a non-digit/letter (i.e., real separator)
    segments = re.findall(r'(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)_', path.split('/')[-1])
    
    positions = []
    for m in segments:
        try:
            coords = [float(x) for x in m]
            positions.append(coords)
        except ValueError:
            print(f"Invalid coordinates found: {m}")
    
    return positions

def get_g_factor_from_path(path: str):
    g_factor = [None]
    match = _g_factor_pattern.search(path)
    if match:
        g_factor.append(float(match.group(1)))
    return g_factor

def get_speakers_from_path(path):
    return re.findall(r'(?<=_)\d+[a-z][a-z0-9]+(?=[-_])', path.split('/')[-1])

def get_speaker_doa_from_path(path):
    return [tuple(float(s) for s in p[:2]) for p in get_speaker_positions_from_path(path)]

def get_dataloader(cfg, root_dir, shuffle=False, transform=None):
    dataset = CLDataset(cfg, root_dir, transform=transform)
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle, num_workers=cfg.num_workers, pin_memory=True, persistent_workers=False)

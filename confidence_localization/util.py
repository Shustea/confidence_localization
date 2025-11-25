import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch.jit import script

def save_sample_as_image(tensor: torch.Tensor, label: torch.Tensor, bound: torch.Tensor, filename: str, title=None, path='/workspaces/confidence_localization/samples/'):
    # Ensure tensor is on CPU and detach if it's a computation graph tensor
    tensor = tensor.squeeze()
    
    tensor = tensor.cpu()
    bound = bound.cpu()

    tensor = tensor.detach()
    bound = bound.detach()

    plt.figure()
    plt.plot(label, torch.rad2deg(torch.abs(tensor - label)), label='error', color='red')
    plt.xlabel('Actual DOA [radians]')
    plt.ylabel('Error At Direction [deg]')
    plt.suptitle('Angle error at target DOA')
    plt.title(title)
    plt.legend()
    plt.grid()

    # Save the image
    plt.savefig(path + 'error_plot_of_' + filename, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()

    time_axis = np.arange(label.shape[-1])

    plt.figure()
    plt.plot(time_axis, tensor, label='Estimation', color='blue')
    plt.plot(time_axis, tensor + bound, label='Estimation Bound', color='cyan')
    plt.plot(time_axis, tensor - bound, label='Estimation Bound', color='cyan')
    plt.plot(time_axis, label, label='GT', color='red')
    plt.suptitle('Estimation angle (with bounds) compared to Ground Truth')
    plt.title(title)
    plt.xlabel('Time [frames]')
    plt.ylabel('Azimuth [radians]')
    plt.legend()

    # Save the image
    plt.savefig(path + 'azimuth_plot_of_'+ filename, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()

    plt.figure()
    plt.plot(time_axis, torch.rad2deg(bound), label='Estimation Bounds', color='black')
    plt.suptitle('Estimation bounds compared to time')
    plt.title(title)
    plt.xlabel('Time [frames]')
    plt.ylabel('error bound [degrees]')
    plt.legend()

    # Save the image
    plt.savefig(path + 'bound_plot_of_'+ filename, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()

@script
def gevd(Rs: torch.Tensor, Rv: torch.Tensor, eps : float = 1e-9) -> torch.Tensor:
    Rv_inv = torch.linalg.inv(Rv)
    L, U = torch.linalg.eig(Rv_inv @ Rs)
    idx = torch.argmax(L.real, dim=0)
    principal_vec = U[:, idx]
    return principal_vec[1:] / (principal_vec[0] + eps)

@script
def cholesky(Rs: torch.Tensor, Rv: torch.Tensor, eps : float = 1e-9) -> torch.Tensor:
    L = torch.linalg.cholesky(Rv)
    L_inv = torch.inverse(L)

    Rs_white = L_inv @ Rs @ L_inv.conj().T

    vals, vecs = torch.linalg.eig(Rs_white)

    idx = torch.argmax(vals.real)
    principal_vec = vecs[:, idx]

    white_vec = L.conj().T @ principal_vec

    return white_vec[1:] / (white_vec[0] + eps)

def energy_vad(x, fs, frame_ms=20, hop_ms=10, alpha=4, win_sec=1):
    frame = int(frame_ms*fs/1000)
    hop   = int(hop_ms*fs/1000)
    W     = int(win_sec*1000/hop_ms)
    # 1. short-time power
    frames = x.unfold(0, frame, hop)
    E      = (frames**2).mean(-1)
    # 2-3. adaptive noise floor + threshold
    pad_E  = torch.cat([E[:1].repeat(W), E])
    noise  = torch.minimum.accumulate(pad_E)[:-W]        # causal min-tracker
    vad    = E > alpha*noise
    # 4-5. 150 ms hang-over
    hang   = int(0.15*1000/hop_ms)
    for i in range(1, hang): vad[:-i] |= vad[i:]
    return vad

def estimate_cov_batched(X):
    return (X @ X.conj().transpose(-2, -1)) / X.shape[-1]

def estimate_rtf(cfg, spectrums, win_len=4):
    M, F_bins, T = spectrums.shape
    T_noise = int((cfg.pre_speech_noise_time * cfg.fs) / (cfg.win_len * (1 - cfg.overlap)))
    win_size = 2 * win_len + 1

    noise_slice = spectrums[..., :T_noise]
    spectrums = spectrums[..., T_noise:]
    # the first second of each sample is strictly noise so we have good bases for our decomposition

    rtf = torch.empty(M - 1, F_bins, spectrums.shape[-1], dtype=torch.complex64, device=spectrums.device)

    for f in range(F_bins):        # (M, T_vad)
        Rv = estimate_cov_batched(noise_slice[:, f])

        padded = F.pad(spectrums[:, f], pad=(win_len, win_len), mode='constant', value=0)  # (M, T + 2w)
        Xf = padded.unfold(-1, size=win_size, step=1)  # (M, T, win_size)
        Xf = Xf.permute(1, 0, 2)  # (T, M, win_size)

        Rs = estimate_cov_batched(Xf)  # (T, M, M)

        for t in range(T-T_noise):
            rtf[:, f, t] = gevd(Rs[t], Rv)
    return rtf  # (M-1, F, T)


def compute_multichannel_stft(signal: np.ndarray, cfg):
    T, _, M = signal.shape
    signal_tensor = torch.from_numpy(signal).float()
    signal_tensor = signal_tensor.permute(2, 0, 1).squeeze(-1)

    stft_list = []
    for mic_idx in range(M):
        stft_mic = torch.stft(
            signal_tensor[mic_idx],
            n_fft=cfg.win_len,
            hop_length=int(cfg.win_len * (1 - cfg.overlap)),
            return_complex=True
        )
        stft_list.append(stft_mic)

    stft_tensor = torch.stack(stft_list, dim=0)
    return stft_tensor

def compute_multichannel_istft(signal: np.ndarray, cfg):
    T, _, M = signal.shape
    signal_tensor = torch.from_numpy(signal).float()
    signal_tensor = signal_tensor.permute(2, 0, 1).squeeze(-1)

    stft_list = []
    for mic_idx in range(M):
        stft_mic = torch.istft(
            signal_tensor[mic_idx],
            n_fft=cfg.win_len,
            hop_length=int(cfg.win_len * (1 - cfg.overlap)),
            return_complex=True
        )
        stft_list.append(stft_mic)

    stft_tensor = torch.stack(stft_list, dim=0)
    return stft_tensor

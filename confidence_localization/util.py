import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eig
import torch.nn.functional as F
from torch.jit import script

def save_sample_as_image(tensor: torch.Tensor, label: torch.Tensor, bound: torch.Tensor, filename: str, title=None, path='/workspaces/confidence_localization/samples/'):
    # Ensure tensor is on CPU and detach if it's a computation graph tensor
    tensor = tensor.squeeze()
    
    tensor = tensor.cpu()
    bound = bound.cpu()

    tensor = tensor.detach()
    bound = bound.detach()

    time_axis = np.arange(label.shape[-1])

    plt.figure()
    plt.plot(time_axis, torch.rad2deg(torch.abs(tensor - label)), label='error', color='red')
    plt.xlabel('Actual DOA [radians]')
    plt.ylabel('Error At Direction [deg]')
    plt.suptitle('Angle error at target DOA')
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # Save the image
    plt.savefig(path + 'error_plot_of_' + filename, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()

    plt.figure()
    plt.plot(time_axis, tensor, label='Estimation', color='blue')
    plt.plot(time_axis, tensor + bound, label='Estimation Bound', color='cyan')
    plt.plot(time_axis, tensor - bound, color='cyan')
    plt.plot(time_axis, label, label='GT', color='red')
    plt.suptitle('Estimation angle (with bounds) compared to Ground Truth')
    plt.title(title)
    plt.xlabel('Time [frames]')
    plt.ylabel('Azimuth [radians]')
    plt.legend()
    plt.grid(True)

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
    plt.grid(True)

    # Save the image
    plt.savefig(path + 'bound_plot_of_'+ filename, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()

def gevd(Rs: torch.Tensor, Rv: torch.Tensor, eps : float = 1e-9) -> torch.Tensor:

    # torch doesn't have gevd solver so well whiten the cov matrices
    L = torch.linalg.cholesky(Rv)
    L_inv = torch.linalg.inv(L)
    Rs_white = L_inv @ Rs @ L_inv.conj().T

    vals, vecs = torch.linalg.eigh(Rs_white)
    idx = torch.argmax(vals.abs())
    
    principal_vec = vecs[:, idx]

    # for added stability
    den = principal_vec[0]
    if abs(den) < 1e-6:
        den = 1e-6 * torch.exp(1j * torch.angle(den))

    return principal_vec / den

@script
def cholesky(Rs: torch.Tensor, Rv: torch.Tensor, eps : float = 1e-9) -> torch.Tensor:
    L = torch.linalg.cholesky(Rv)
    L_inv = torch.linalg.inv(L)

    Rs_white = L_inv @ Rs @ L_inv.conj().T # now the signal covariance is white

    vals, vecs = torch.linalg.eigh(Rs_white) 

    idx = torch.argmax(vals.abs())
    principal_vec = vecs[:, idx]

    white_vec = L_inv.conj().T @ principal_vec

    v = white_vec / (white_vec[0] + eps)
    return v


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

def estimate_rtf(cfg, spectrums, epsilon=0.01):
    M, F, T = spectrums.shape
    device, cdtype = spectrums.device, spectrums.dtype

    K = min(F, cfg.nfft // 2 + 1)
    hop = (1.0 - cfg.overlap) * cfg.nfft
    Tn = int(np.ceil(cfg.pre_speech_noise_time * cfg.fs / hop))
    Tn = max(0, min(Tn, T - 1))

    z_n = spectrums[:, :K, :Tn]
    z_s = spectrums[:, :K, Tn:]
    Ts = z_s.shape[-1]

    Rv = torch.zeros((K, M, M), dtype=cdtype, device=device)
    L  = torch.zeros((K, M, M), dtype=cdtype, device=device)

    I = torch.eye(M, device=device, dtype=cdtype)

    Xn = z_n.permute(1, 0, 2)
    Rv = (Xn @ Xn.conj().transpose(-1, -2)) / Xn.shape[-1]
    Rv = Rv + (epsilon * torch.linalg.norm(Rv, ord="fro", dim=(-2, -1)).view(K, 1, 1)) * I
    L = torch.linalg.cholesky((Rv + Rv.conj().permute(0,-1,-2))/2)

    alpha = cfg.exp_window_smoothing
    win_len = cfg.win_len

    Rs_prev = torch.zeros((K, M, M), dtype=cdtype, device=device)
    G_tf = torch.zeros((K, Ts, M), dtype=cdtype, device=device)

    for k in range(K):
        Lk = L[k]
        for t in range(Ts):
            a = max(0, t - win_len)
            b = min(Ts - 1, t + win_len)
            x = z_s[:, k, a:b+1]
            Xw = torch.linalg.solve_triangular(Lk, x, upper=False)
            Rs = (Xw @ Xw.conj().T) / Xw.shape[1]
            if t > 0:
                Rs = alpha * Rs_prev[k] + (1 - alpha) * Rs
            Rs_prev[k] = Rs
            w, V = torch.linalg.eig(Rs)
            i = w.abs().argmax()
            psi = V[:, i]
            u = Lk @ psi
            g = u / u[0]
            G_tf[k, t] = g

    absG = G_tf.abs()
    thr = 3 * absG.mean(dim=0, keepdim=True)
    mask = absG > thr
    G_tf = G_tf.clone()
    G_tf[mask] = (torch.randint(0, 2, (mask.sum(),), device=G_tf.device) * 2 - 1).to(G_tf.dtype)

    reir_matrix = torch.zeros((2 * (K - 1), Ts, M - 1), dtype=torch.float32, device=device)
    for m in range(1, M):
        Gm = G_tf[:, :, m]
        reir_matrix[:, :, m - 1] = torch.fft.irfft(Gm, dim=0).to(torch.float32)

    return torch.fft.ifftshift(reir_matrix, dim=0).permute(2,1,0)

def rtf_to_reir(rtf, F_L=None, F_R=None):
    M, K, T = rtf.shape
    rtf_np = rtf.cpu().numpy()
    nfft = 2*(K-1)
    if F_L is None: F_L = K
    if F_R is None: F_R = K
    reir = np.zeros((M, nfft, T), dtype=np.float32)

    for t in range(T):
        for m in range(M):
            G = rtf_np[m, :, t]
            G_full = np.concatenate([G, G[1:-1].conj()[::-1]])
            g = np.fft.ifft(G_full)
            out = np.zeros_like(g.real)
            out[:F_R] = g.real[:F_R]
            out[-F_L:] = g.real[-F_L:]
            reir[m, :, t] = out

    return torch.from_numpy(reir)

def compute_multichannel_stft(signal: np.ndarray, cfg):
    """
    compute stft for every channel across a file - make sure the input is channels x time
    """
    M, T = signal.shape
    if type(signal) != torch.Tensor:
        signal_tensor = torch.tensor(signal)
    else:
        signal_tensor = signal
    window = torch.hamming_window(cfg.nfft, periodic=True)

    stft_list = []
    for mic_idx in range(M):
        stft_mic = torch.stft(
            signal_tensor[mic_idx],
            n_fft=cfg.nfft,
            hop_length=int(cfg.nfft * (1 - cfg.overlap)),
            return_complex=True,
            window=window
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

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.linalg import eig
import torch.nn.functional as F
from torch.jit import script

def save_sample_as_image(
    tensor: torch.Tensor,
    label: torch.Tensor,
    bound: torch.Tensor,
    filename: str,
    spectrum: torch.Tensor = None,
    title: str = None,
    path: str = '/workspaces/confidence_localization/samples/',
    waveform: torch.Tensor = None,
    fs: int = None,
    mic_positions=None,
    source_radius: float = None,
    room_dim=None,
):
    """Save a publication-quality DOA evaluation figure.

    Optional scenario panel (top): top-down xy view of mic array + source
    trajectory when ``mic_positions`` and ``source_radius`` are supplied (and
    ``room_dim`` if you also want the room outline). Optional WAV panel:
    channel-0 waveform when ``waveform`` (and ``fs``) are supplied. Required
    panels: azimuth estimate with ±1σ confidence band vs. ground truth;
    absolute angular error and predicted confidence bound. Optional REIR slice
    (bottom): 1D REIR at the middle time frame, with one line per channel,
    when ``spectrum`` has shape ``[C, T, Freq/Lag]``.
    """
    tensor = tensor.squeeze().detach().cpu().float()
    label = label.squeeze().detach().cpu().float()
    bound = bound.squeeze().detach().cpu().float()

    doa_deg = torch.rad2deg(tensor).numpy()
    bound_deg = torch.rad2deg(bound).numpy()

    valid = torch.isfinite(label)
    valid_np = valid.numpy()
    label_deg = np.where(valid_np, np.rad2deg(label.numpy()), np.nan)

    finite_idx = np.flatnonzero(valid_np)
    if finite_idx.size > 0:
        start_doa = float(label[int(finite_idx[0])])
        end_doa = float(label[int(finite_idx[-1])])
        panel_title = f"starting DOA {start_doa:.4f} rad → ending DOA {end_doa:.4f} rad"
    else:
        panel_title = "starting DOA (n/a) → ending DOA (n/a)"

    diff = torch.remainder(tensor - label + torch.pi, 2 * torch.pi) - torch.pi
    error_deg = np.where(valid_np, torch.rad2deg(diff.abs()).numpy(), np.nan)

    t = np.arange(doa_deg.shape[0])
    _BLUE = '#2166ac'
    _RED = '#d6604d'

    rc = {
        'font.family': 'serif',
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.color': '#cccccc',
        'grid.linestyle': '--',
        'grid.linewidth': 0.6,
        'lines.linewidth': 1.5,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
    }

    reir = None
    if spectrum is not None:
        reir = spectrum.detach().cpu().float().numpy()
        if reir.ndim != 3:
            reir = None

    wav_np = None
    if waveform is not None:
        wav_t = waveform.detach().cpu().float().squeeze()
        if wav_t.ndim > 1:
            wav_t = wav_t[0]
        if wav_t.numel() > 1:
            wav_np = wav_t.numpy()

    has_reir = reir is not None
    has_wav = wav_np is not None

    # Top-down scenario: mic array + interpolated source trajectory on xy plane.
    scenario = None
    if mic_positions is not None and source_radius is not None:
        mic_arr = np.asarray(mic_positions, dtype=np.float64)
        if mic_arr.ndim == 2 and mic_arr.shape[1] >= 2:
            center_xy = mic_arr[:, :2].mean(axis=0)
            labels_np = label.numpy().reshape(-1)
            valid_mask = np.isfinite(labels_np)
            n_valid = int(valid_mask.sum())
            n_total = int(labels_np.shape[0])
            traj_xy = start_xy = end_xy = None
            if n_valid >= 2:
                first_idx = int(np.argmax(valid_mask))
                last_idx = int(n_total - 1 - np.argmax(valid_mask[::-1]))
                az_unwrapped = np.unwrap(labels_np[first_idx:last_idx + 1])
                az_full = np.full(n_total, np.nan, dtype=np.float64)
                az_full[first_idx:last_idx + 1] = az_unwrapped
                idx_v = np.where(np.isfinite(az_full))[0]
                r = float(source_radius)
                traj_xy = np.empty((idx_v.size, 2), dtype=np.float64)
                traj_xy[:, 0] = center_xy[0] + r * np.cos(az_full[idx_v])
                traj_xy[:, 1] = center_xy[1] + r * np.sin(az_full[idx_v])
                start_xy = (float(traj_xy[0, 0]), float(traj_xy[0, 1]))
                end_xy = (float(traj_xy[-1, 0]), float(traj_xy[-1, 1]))
            scenario = {
                'mic_xy': mic_arr[:, :2],
                'center_xy': center_xy,
                'traj_xy': traj_xy,
                'start_xy': start_xy,
                'end_xy': end_xy,
                'n_valid': n_valid,
                'n_total': n_total,
                'source_radius': float(source_radius),
                'start_az': float(labels_np[int(np.argmax(valid_mask))]) if n_valid >= 1 else None,
                'end_az': float(labels_np[int(n_total - 1 - np.argmax(valid_mask[::-1]))]) if n_valid >= 1 else None,
            }
    has_scenario = scenario is not None

    # Build panel layout: optional scenario (top), optional wav, required
    # azimuth+error, optional REIR.
    height_ratios = []
    if has_scenario:
        height_ratios.append(4)            # square-ish for aspect='equal'
    if has_wav:
        height_ratios.append(2)
    height_ratios.extend([3, 2])           # azimuth, error
    if has_reir:
        height_ratios.append(2)
    n_rows = len(height_ratios)
    fig_h = 1.6 * n_rows + 1.2

    with plt.rc_context(rc):
        fig, axes = plt.subplots(
            n_rows, 1, figsize=(8, fig_h),
            gridspec_kw={'height_ratios': height_ratios},
        )
        if n_rows == 1:
            axes = [axes]
        fig.subplots_adjust(hspace=0.35)
        fig.suptitle(filename, fontsize=12, y=0.995)

        # Resolve panel index map.
        idx = 0
        if has_scenario:
            ax_scn = axes[idx]; idx += 1
            mic_xy = scenario['mic_xy']
            center_xy = scenario['center_xy']
            traj_xy = scenario['traj_xy']
            r = scenario['source_radius']
            if room_dim is not None:
                rd = np.asarray(room_dim, dtype=float).reshape(-1)
                if rd.size >= 2:
                    x1, y1 = float(rd[0]), float(rd[1])
                    ax_scn.plot(
                        [0, x1, x1, 0, 0], [0, 0, y1, y1, 0],
                        lw=0.8, color='#888888', alpha=0.7, label='Room',
                    )
            theta_ref = np.linspace(0, 2 * np.pi, 256)
            ax_scn.plot(
                center_xy[0] + r * np.cos(theta_ref),
                center_xy[1] + r * np.sin(theta_ref),
                ls=':', color='#888888', lw=0.6, alpha=0.6,
            )
            if traj_xy is not None:
                step = max(1, traj_xy.shape[0] // 600)
                ax_scn.plot(
                    traj_xy[::step, 0], traj_xy[::step, 1], '.',
                    color=_BLUE, ms=3.0, alpha=0.6, label='Source trajectory',
                )
                ax_scn.scatter(
                    *scenario['start_xy'], color='#D95319', s=70, zorder=5,
                    edgecolors='k', linewidths=0.5,
                    label=f"Start ({np.rad2deg(scenario['start_az']):.0f}°)",
                )
                ax_scn.scatter(
                    *scenario['end_xy'], color='#EDB120', s=110, marker='*',
                    zorder=5, edgecolors='k', linewidths=0.3,
                    label=f"End ({np.rad2deg(scenario['end_az']):.0f}°)",
                )
            ax_scn.scatter(
                mic_xy[:, 0], mic_xy[:, 1], color='#7E2F8E', s=80, marker='^',
                zorder=5, edgecolors='k', linewidths=0.5, label='Mics',
            )
            ax_scn.set_xlabel('x (m)')
            ax_scn.set_ylabel('y (m)')
            ax_scn.set_aspect('equal', adjustable='box')
            ax_scn.set_title(
                f"Scenario (top-down)  —  {scenario['n_valid']}/{scenario['n_total']} frames valid",
                pad=4,
            )
            ax_scn.legend(
                loc='upper right', framealpha=0.9, edgecolor='#aaaaaa', fontsize=8,
            )

        if has_wav:
            ax_wav = axes[idx]; idx += 1
            fs_safe = max(1, int(fs)) if fs else 16000
            t_wav = np.arange(wav_np.shape[0]) / fs_safe
            ax_wav.plot(t_wav, wav_np, color='#333333', lw=0.6)
            ax_wav.set_ylabel('ch0')
            ax_wav.set_title(
                f'waveform ch0  (fs={fs_safe} Hz, {wav_np.shape[0]} samples)', pad=4,
            )
            ax_wav.set_xlim(t_wav[0], t_wav[-1] if t_wav.size > 1 else 1.0)
            ax_wav.tick_params(labelbottom=True)

        ax1, ax2 = axes[idx], axes[idx + 1]
        ax1.sharex(ax2)

        ax1.fill_between(t, doa_deg - bound_deg, doa_deg + bound_deg,
                         color=_BLUE, alpha=0.18, label='±1σ confidence', zorder=2)
        ax1.plot(t, doa_deg, color=_BLUE, lw=1.5, label='Estimate', zorder=3)
        ax1.plot(t, label_deg, color=_RED, lw=1.5, ls='--', label='Ground truth', zorder=4)
        ax1.set_ylabel('Azimuth (°)')
        all_vals = np.concatenate([doa_deg, label_deg[~np.isnan(label_deg)]])
        if all_vals.size > 0:
            lo, hi = float(np.min(all_vals)), float(np.max(all_vals))
            pad = max(5.0, 0.1 * (hi - lo))
            ax1.set_ylim(lo - pad, hi + pad)
        ax1.legend(loc='upper right', framealpha=0.9, edgecolor='#aaaaaa')
        ax1.set_title(panel_title, pad=5)

        ax2.fill_between(t, 0, bound_deg, color=_BLUE, alpha=0.18, zorder=2)
        ax2.plot(t, bound_deg, color=_BLUE, lw=1.2, ls='--', label='σ bound', zorder=3)
        ax2.plot(t, error_deg, color=_RED, lw=1.2, label='|Error|', zorder=4)
        ax2.set_ylabel('Error (°)')
        ax2.set_xlabel('Time (frames)')
        ax2.set_ylim(bottom=0)
        ax2.legend(loc='upper right', framealpha=0.9, edgecolor='#aaaaaa')

        if has_reir:
            ax3 = axes[idx + 2]
            C, T_reir, L = reir.shape
            t_mid = T_reir // 2
            lag_axis = np.arange(L) - L // 2
            cmap = plt.get_cmap('viridis')
            for c in range(C):
                ax3.plot(
                    lag_axis, reir[c, t_mid, :],
                    color=cmap(c / max(1, C - 1)),
                    lw=1.2, label=f'ch {c + 1}',
                )
            ax3.axvline(0, color='#888888', lw=0.6, ls=':')
            ax3.set_xlabel('Lag (samples)')
            ax3.set_ylabel('REIR')
            ax3.set_title(f'REIR slice @ t = {t_mid}', pad=5)
            ax3.legend(loc='upper right', framealpha=0.9, edgecolor='#aaaaaa', ncol=C)

        out = Path(path) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches='tight')
        plt.close(fig)

def save_room_geometry(
    labels,
    room_dim,
    mic_positions,
    source_radius: float,
    src_height: float,
    filename: str = "room_geometry_example.png",
    title: str = "Room Geometry",
    path: str = "/workspaces/confidence_localization/samples/",
) -> None:
    """Save a 3D room-geometry plot mirroring data/signal_generator/run_example.py.

    The source XY is reconstructed from ``labels`` (per-frame azimuth in rad) on
    a fixed circle at ``source_radius`` around the microphone-array centroid and
    placed at ``src_height``. NaN frames are dropped; if fewer than 2 valid
    frames remain, the trajectory is suppressed but the room + mics still render.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers projection='3d')

    labels_np = labels.detach().cpu().float().numpy() if torch.is_tensor(labels) else np.asarray(labels, dtype=np.float32)
    labels_np = labels_np.reshape(-1)
    n_total = int(labels_np.shape[0])
    valid_mask = np.isfinite(labels_np)
    n_valid = int(valid_mask.sum())

    room_dim_arr = np.asarray(room_dim, dtype=np.float64).reshape(-1)
    if room_dim_arr.size != 3:
        raise ValueError(f"room_dim must have 3 entries, got {tuple(room_dim_arr.shape)}")
    mic_arr = np.asarray(mic_positions, dtype=np.float64)
    if mic_arr.ndim != 2 or mic_arr.shape[1] != 3:
        raise ValueError(f"mic_positions must be [M, 3], got {tuple(mic_arr.shape)}")
    array_center = mic_arr.mean(axis=0)

    if n_valid >= 2:
        first_idx = int(np.argmax(valid_mask))
        last_idx = int(n_total - 1 - np.argmax(valid_mask[::-1]))
        start_az = float(labels_np[first_idx])
        end_az = float(labels_np[last_idx])
        # Interpolate azimuth across the recording (unwrapped) so the trajectory
        # is a smooth arc between start and end, matching the reference plot.
        az_unwrapped = np.unwrap(labels_np[first_idx:last_idx + 1])
        az_full = np.full(n_total, np.nan, dtype=np.float64)
        az_full[first_idx:last_idx + 1] = az_unwrapped
        valid_indices = np.where(np.isfinite(az_full))[0]
        sp_path = np.empty((valid_indices.size, 3), dtype=np.float64)
        sp_path[:, 0] = array_center[0] + source_radius * np.cos(az_full[valid_indices])
        sp_path[:, 1] = array_center[1] + source_radius * np.sin(az_full[valid_indices])
        sp_path[:, 2] = src_height
        suffix = f"({n_valid}/{n_total} frames valid)"
    else:
        sp_path = None
        start_az = end_az = None
        suffix = f"({n_valid}/{n_total} frames valid — trajectory unavailable)"

    # MATLAB-default palette to match run_example.py.
    palette = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E"]

    rc = {
        "font.family": "serif", "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 10,
        "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }

    with plt.rc_context(rc):
        fig = plt.figure(figsize=(7.5, 5.5))
        ax = fig.add_subplot(111, projection="3d")

        # Wireframe box: 12 edges of [0, room_dim] cube.
        x0, y0, z0 = 0.0, 0.0, 0.0
        x1, y1, z1 = float(room_dim_arr[0]), float(room_dim_arr[1]), float(room_dim_arr[2])
        corners = np.array([
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ])
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        for a, b in edges:
            ax.plot(
                [corners[a, 0], corners[b, 0]],
                [corners[a, 1], corners[b, 1]],
                [corners[a, 2], corners[b, 2]],
                lw=0.6, color="#aaaaaa", alpha=0.4,
            )

        if sp_path is not None:
            step = max(1, sp_path.shape[0] // 600)
            ax.plot(
                sp_path[::step, 0], sp_path[::step, 1], sp_path[::step, 2],
                ".", color=palette[0], ms=3.0, alpha=0.55,
                label="Source trajectory", zorder=2,
            )
            ax.scatter(
                *sp_path[0], color=palette[1], s=80, zorder=5,
                edgecolors="k", linewidths=0.5,
                label=f"Start ({np.rad2deg(start_az):.0f}°)",
            )
            ax.scatter(
                *sp_path[-1], color=palette[2], s=100, marker="*", zorder=5,
                edgecolors="k", linewidths=0.3,
                label=f"End ({np.rad2deg(end_az):.0f}°)",
            )
        ax.scatter(
            mic_arr[:, 0], mic_arr[:, 1], mic_arr[:, 2],
            color=palette[3], s=100, marker="^", zorder=5,
            edgecolors="k", linewidths=0.5, label="Microphones",
        )

        ax.set_xlim(0, x1); ax.set_xlabel("x (m)")
        ax.set_ylim(0, y1); ax.set_ylabel("y (m)")
        ax.set_zlim(0, z1); ax.set_zlabel("z (m)")
        ax.set_title(f"{title}  {suffix}")
        ax.view_init(elev=25, azim=-50)
        ax.legend(loc="upper left", fontsize=10, framealpha=0.85)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        out = Path(path) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)


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


def energy_vad(x, fs, frame_ms=20, hop_ms=10, alpha=4, win_sec=1, hangover_ms=150):
    """Energy-based VAD with adaptive noise floor + hangover smoothing.

    Input: 1D waveform tensor ``x`` at sample rate ``fs``.
    Output: 1D bool tensor of length ``(len(x) - frame) // hop + 1`` with ``True`` for speech frames.
    """
    if x.ndim > 1:
        x = x.reshape(-1)
    frame = int(frame_ms * fs / 1000)
    hop = int(hop_ms * fs / 1000)
    W = int(win_sec * 1000 / hop_ms)
    frames = x.unfold(0, frame, hop)
    E = (frames ** 2).mean(-1)
    pad_E = torch.cat([E[:1].repeat(W), E])
    noise = torch.cummin(pad_E, dim=0).values[:-W]
    vad = E > alpha * noise
    hang = max(1, int(hangover_ms / hop_ms))
    for i in range(1, hang):
        tail = vad[i:].clone()
        vad[:-i] = vad[:-i] | tail
    return vad


def vad_to_rtf_frames(vad, vad_hop_ms, rtf_hop_samples, fs, n_rtf_frames):
    """Nearest-neighbor resample a VAD bool tensor onto the RTF frame grid.

    ``vad``: bool tensor at ``vad_hop_ms`` hop.
    Returns a bool tensor of length ``n_rtf_frames`` aligned to the RTF frames.
    """
    rtf_hop_ms = 1000.0 * rtf_hop_samples / fs
    src_idx = (np.arange(n_rtf_frames) * rtf_hop_ms / vad_hop_ms).astype(np.int64)
    src_idx = np.clip(src_idx, 0, vad.shape[0] - 1)
    vad_np = vad.cpu().numpy() if torch.is_tensor(vad) else np.asarray(vad)
    return torch.from_numpy(vad_np[src_idx].astype(bool))


def estimate_cov_batched(X):
    return (X @ X.conj().transpose(-2, -1)) / X.shape[-1]

def estimate_rtf(cfg, spectrums, epsilon=0.01):
    """Estimate RTF features from multichannel STFT data (batched eig per-frame).

    Input: ``spectrums`` with shape ``[M, F, T]``.
    Output: an RTF feature tensor with shape ``[M - 1, Ts, 2 * (K - 1)]``.
    """
    M, F, T = spectrums.shape
    device, cdtype = spectrums.device, spectrums.dtype

    K = min(F, cfg.nfft // 2 + 1)
    hop = (1.0 - cfg.overlap) * cfg.nfft
    Tn = int(np.ceil(cfg.pre_speech_noise_time * cfg.fs / hop))
    Tn = max(0, min(Tn, T - 1))

    z_n = spectrums[:, :K, :Tn]
    z_s = spectrums[:, :K, Tn:]
    Ts = z_s.shape[-1]

    I = torch.eye(M, device=device, dtype=cdtype)

    if Tn > 0:
        Xn = z_n.permute(1, 0, 2)
        Rv = (Xn @ Xn.conj().transpose(-1, -2)) / Xn.shape[-1]
    else:
        Rv = epsilon * I.unsqueeze(0).expand(K, -1, -1).clone()

    # Symmetrize and Cholesky-factorize. Use UNREGULARIZED L for the back-transform
    # and a SEPARATE regularized L_reg for the whitening (matches MATLAB convention).
    Rv_sym = (Rv + Rv.conj().permute(0, -1, -2)) / 2
    Rv_diag = torch.diagonal(Rv_sym, dim1=-2, dim2=-1).real.mean(dim=-1).clamp(min=1e-10)
    Rv_sym = Rv_sym + (epsilon * Rv_diag).view(K, 1, 1) * I
    L = torch.linalg.cholesky(Rv_sym)

    L_norms = torch.linalg.norm(L, dim=(-2, -1)).view(K, 1, 1)
    L_reg = L + epsilon * L_norms * I

    win_len = cfg.win_len
    zs_kmt = z_s.permute(1, 0, 2)
    G_tf = torch.zeros((K, Ts, M), dtype=cdtype, device=device)

    for t in range(Ts):
        a = max(0, t - win_len)
        b = min(Ts - 1, t + win_len)
        ws = b - a + 1
        x_batch = zs_kmt[:, :, a:b+1]
        Xw = torch.linalg.solve_triangular(L_reg, x_batch, upper=False)
        Rs = (Xw @ Xw.conj().transpose(-1, -2)) / ws
        _, V = torch.linalg.eigh(Rs)
        psi = V[:, :, -1]
        u = (L @ psi.unsqueeze(-1)).squeeze(-1)
        g = u / (u[:, 0:1] + 1e-8)

        # Outlier replacement per-frame per-channel
        for m in range(1, M):
            col = g[:, m]
            thr = 3.0 * col.abs().mean()
            mask = col.abs() > thr
            n_out = mask.sum().item()
            if n_out > 0:
                signs = 2.0 * torch.bernoulli(0.5 * torch.ones(n_out, device=device)) - 1.0
                g[mask, m] = signs.to(cdtype)

        G_tf[:, t, :] = g

    # No spatial info at DC; Nyquist often noisy
    G_tf[0, :, :] = 0
    G_tf[-1, :, :] = 0

    reir_matrix = torch.fft.irfft(G_tf[:, :, 1:], dim=0).to(torch.float32)
    return torch.fft.ifftshift(reir_matrix, dim=0).permute(2, 1, 0)

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

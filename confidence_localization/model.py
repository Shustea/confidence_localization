import math

import torch
import torch.nn.functional as F
from scipy.special import i0e

## Loss functions ##

TAU_68 = 0.6827

def ang_wrap(x):
    """Wrap angular values into the principal interval ``[-pi, pi]``."""
    return (x + torch.pi) % (2 * torch.pi) - torch.pi


def kappa_to_circ_std(kappa, eps=1e-12):
    """Convert Von Mises concentration to circular standard deviation."""
    # circular variance = 1 - I1(kappa)/I0(kappa), approximated via A(kappa) = I1/I0
    # For numerical stability use the ratio of Bessel functions
    A = 1.0 - 1.0 / (2.0 * kappa.clamp(min=eps))  # large-kappa approx of I1/I0
    # More accurate: use torch.special.i1e / torch.special.i0e (both already scaled by exp(-|k|))
    i0e = torch.special.i0e(kappa)
    i1e = torch.special.i1e(kappa)
    A = i1e / (i0e + eps)
    circ_var = (1.0 - A).clamp(min=eps)
    return torch.sqrt(-2.0 * torch.log(1.0 - circ_var + eps))

def ang_err_from_unit(doa_unit, labels):
    """Compute the signed angular error between unit-vector predictions and angle labels."""
    mu = torch.atan2(doa_unit[..., 1], doa_unit[..., 0])
    d = ang_wrap(labels - mu)
    return d

def weights(labels, vad=None):
    """Build a validity mask from finite labels and optionally combine it with a VAD mask."""
    w = torch.isfinite(labels).float()
    return w if vad is None else w * vad.float()

def safe_denom(w):
    """Return a denominator that is clamped away from zero for stable normalization."""
    return w.sum().clamp_min(1.0)

def pinball(u, tau):
    """Compute the asymmetric pinball residual used for quantile-style losses."""
    return torch.maximum(tau * u, (tau - 1.0) * u)

def soft_indicator(x, tau):
    """Approximate a hard indicator function with a temperature-controlled sigmoid."""
    return torch.sigmoid(x / tau)

def gaussian_loss(mean_err_squared, log_std, log_var_weight=1):
    """Compute a Gaussian-style loss from mean squared error and predicted log variance."""
    return (0.5 * (mean_err_squared * torch.exp(-log_std)) + (log_var_weight * log_std)).mean() 

def hetero_gaussian_nll_err(err, log_var, weights=None, min_log_var=-10.0, max_log_var=5.0):
    """Compute a heteroscedastic Gaussian negative log-likelihood from angular errors and log variance."""
    lv = log_var.clamp(min_log_var, max_log_var)
    inv_var = (-lv).exp()
    loss = 0.5 * (err.pow(2) * inv_var + lv)
    if weights is not None: loss = loss * weights
    return loss.mean()

def vm_nll_calibrated(err, kappa_raw, alpha, mask=None, lam=0.1, p=0.68, tau=0.05, kappa_max=200., eps=1e-12):
    """Compute a calibrated von Mises negative log-likelihood with a soft coverage penalty.

    Example:
        Input:
            ``err = tensor([0.05, -0.10, 0.20])``
            ``kappa_raw = tensor([2.0, 2.0, 2.0])``
            ``alpha = tensor(0.0)``
        Output:
            a scalar loss. Smaller absolute angular errors or larger effective
            concentration values reduce the NLL term, while poor 68% coverage
            increases the calibration penalty.
    """
    k = F.softplus(kappa_raw).clamp(eps, kappa_max)
    a = torch.exp(alpha).clamp(1e-3, 1e3)
    k = (a * k).clamp(eps, kappa_max)

    nll = -k*torch.cos(err) + math.log(2*math.pi) + (torch.log(torch.special.i0e(k) + eps) + k)

    sigma = kappa_to_circ_std(k, eps)
    cov = torch.sigmoid((sigma - err.detach().abs()) / tau)  # detach so calib tunes kappa, not mean

    if mask is not None:
        nll = nll * mask
        cov = cov * mask
        denom = mask.sum() + eps
        nll = nll.sum() / denom
        cov = cov.sum() / denom
    else:
        nll = nll.mean()
        cov = cov.mean()

    return nll + lam * (cov - p).pow(2)

def von_mises_loss(mean_err_squared, log_std, log_var_weight=1):
    """Compute a von Mises-style loss from squared angular error and predicted log variance.

    Example:
        Input: ``mean_err_squared`` and ``log_std`` with matching shape ``[B, T]``.
        Output: a scalar mean loss where larger angular error or unrealistically
        small predicted uncertainty increases the penalty.
    """
    angle_error = torch.sqrt(mean_err_squared).clamp(0, torch.pi)
    log_kappa = -2 * log_std
    kappa = torch.exp(log_kappa).clamp(max=5e2).to(torch.float64) # since the float is numerically limited, using a max kappa of 500 will bring us good enough results 

    log_I0 = torch.log(torch.i0(kappa) + 1e-8)
    nll = -(kappa * torch.cos(angle_error) - log_I0) + log_var_weight * log_kappa # von Mises NLL: -log(I0(kappa)) - kappa * cos(error)
    return nll.mean()

def halfnormal_loss(error, bound, weights, sigma_min=1e-3, sigma_max=1e9):
    """Compute a half-normal NLL for an absolute angular error and predicted scale.

    Example:
        Input: ``error``, ``bound``, and ``weights`` with shape ``[B, T]``.
        Output: one scalar loss. If ``bound`` grows while ``error`` stays fixed,
        the quadratic penalty weakens but the log-scale term grows.
    """
    sigma = (F.softplus(bound) + sigma_min).clamp_max(sigma_max)
    nll = torch.log(sigma) + 0.5 * (error / sigma).pow(2)

    return (nll * weights).sum() / weights.sum()


def pinball_loss(error, bound, weights, q=TAU_68):
    """Compute the quantile-regression loss used for uncertainty bounds.

    Example:
        Input: ``error = tensor([0.1, 0.4])``, ``bound = tensor([0.2, 0.2])``.
        Output: the first element contributes a small penalty because the bound
        covers the error, while the second contributes a larger one because the
        error exceeds the predicted bound.
    """

    assert 0.0 < q < 1.0, q
    assert torch.isfinite(weights).all()
    assert (weights >= 0).all(), (weights.min().item(), weights.max().item())

    U = error - bound
    L = torch.maximum(q*U, (q-1)*U)
    return (L * weights).sum() / weights.sum().clamp_min(1.0)


def kappa_loss(error, bound, weights, kappa_min=1e-3, kappa_max=1e3):
    """Compute a von Mises negative log-likelihood from wrapped angular error.

    Example:
        Input: ``error`` and ``bound`` with shape ``[B, T]``.
        Output: a scalar loss where near-zero wrapped error and larger effective
        concentration produce a smaller penalty.
    """
    d = ang_wrap(error)

    kappa = (F.softplus(bound) + kappa_min).clamp_max(kappa_max)
    logI0 = kappa + torch.log(i0e(kappa) + 1e-12)
    
    nll = -kappa * torch.cos(d) + (torch.log(torch.tensor(2.0 * torch.pi, device=kappa.device, dtype=kappa.dtype)) + logI0)
    return (nll * weights).sum() / weights.sum()


def bound_loss(error, bound, weights, p=TAU_68, temp=0.02):
    """Penalize mismatch between predicted bound coverage and the target coverage rate.

    Example:
        Input: framewise ``error`` and predicted ``bound`` tensors.
        Output: a scalar near zero when roughly 68% of the weighted errors fall
        inside the softened bound, and larger otherwise.
    """
    b = F.softplus(bound)
    cover = soft_indicator(b - error, temp)
    c = (cover * weights).sum()

    return (c - p).pow(2)


def bound_loss_rank(error, bound, weights):
    """Encourage larger observed errors to receive larger predicted bounds.

    Example:
        Input: two frames where ``error[1] > error[0]``.
        Output: a smaller loss when ``bound[1]`` is also larger than ``bound[0]``.
    """
    weights = weights.reshape(-1) > 0

    b = F.softplus(bound).reshape(-1)

    error = error.reshape(-1)[weights] 
    b = b[weights]

    p = torch.randperm(error.numel(), device=error.device)
    s = torch.sign(error - error[p])

    return F.softplus(-s * (b - b[p])).mean()

from numpy import ceil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121
from einops import rearrange, repeat, einsum
from scipy.special import i0, i0e

## Loss functions ##

TAU_68 = 0.6827

def ang_wrap(x):
    return (x + torch.pi) % (2 * torch.pi) - torch.pi

def ang_err_from_unit(doa_unit, labels):
    mu = torch.atan2(doa_unit[..., 1], doa_unit[..., 0])
    d = ang_wrap(labels - mu)
    return d

def weights(labels, vad=None):
    w = torch.isfinite(labels).float()
    return w if vad is None else w * vad.float()

def safe_denom(w):
    return w.sum().clamp_min(1.0)

def pinball(u, tau):
    return torch.maximum(tau * u, (tau - 1.0) * u)

def soft_indicator(x, tau):
    return torch.sigmoid(x / tau)

def gaussian_loss(mean_err_squared, log_std, log_var_weight=1):
    return (0.5 * (mean_err_squared * torch.exp(-log_std)) + (log_var_weight * log_std)).mean() 

def hetero_gaussian_nll_err(err, log_var, weights=None, min_log_var=-10.0, max_log_var=5.0):
    lv = log_var.clamp(min_log_var, max_log_var)
    inv_var = (-lv).exp()
    loss = 0.5 * (err.pow(2) * inv_var + lv)
    if weights is not None: loss = loss * weights
    return loss.mean()

import math

def kappa_to_circ_std(kappa, eps=1e-12):
    R = (torch.special.i1e(kappa) / torch.special.i0e(kappa)).clamp(eps, 1-eps)
    return torch.sqrt(-2.0 * torch.log(R))

def vm_nll_calibrated(err, kappa_raw, alpha, mask=None, lam=0.1, p=0.68, tau=0.05, kappa_max=200., eps=1e-12):
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
    angle_error = torch.sqrt(mean_err_squared).clamp(0, torch.pi)
    log_kappa = -2 * log_std
    kappa = torch.exp(log_kappa).clamp(max=5e2).to(torch.float64) # since the float is numerically limited, using a max kappa of 500 will bring us good enough results 

    log_I0 = torch.log(torch.i0(kappa) + 1e-8)
    nll = -(kappa * torch.cos(angle_error) - log_I0) + log_var_weight * log_kappa # von Mises NLL: -log(I0(kappa)) - kappa * cos(error)
    return nll.mean()

def kappa_to_circ_std(kappa, eps=1e-12):
    kappa = torch.clamp(kappa, min=eps)
    R = (torch.special.i1e(kappa) / torch.special.i0e(kappa)).clamp(eps, 1-eps)
    return torch.sqrt(-2.0 * torch.log(R))

def halfnormal_loss(error, bound, weights, sigma_min=1e-3, sigma_max=1e9):
    """
    Used for estimating the DOA as a Wrapped-Gaussian Distribiution
    """
    sigma = (F.softplus(bound) + sigma_min).clamp_max(sigma_max)
    nll = torch.log(sigma) + 0.5 * (error / sigma).pow(2)

    return (nll * weights).sum() / weights.sum()


def pinball_loss(error, bound, weights, q=TAU_68):
    """
    Used for quantile regression
    """

    assert 0.0 < q < 1.0, q
    assert torch.isfinite(weights).all()
    assert (weights >= 0).all(), (weights.min().item(), weights.max().item())

    U = error - bound
    L = torch.maximum(q*U, (q-1)*U)
    return (L * weights).sum() / weights.sum().clamp_min(1.0)


def kappa_loss(error, bound, weights, kappa_min=1e-3, kappa_max=1e3):
    """
    Used for estimating the DOA as a Von-Misus Distribiution
    """
    d = ang_wrap(error)

    kappa = (F.softplus(bound) + kappa_min).clamp_max(kappa_max)
    logI0 = kappa + torch.log(i0e(kappa) + 1e-12)
    
    nll = -kappa * torch.cos(d) + (torch.log(torch.tensor(2.0 * torch.pi, device=kappa.device, dtype=kappa.dtype)) + logI0)
    return (nll * weights).sum() / weights.sum()


def bound_loss(error, bound, weights, p=TAU_68, temp=0.02):
    """
    Used to approximate the upper-bound of the error
    """
    b = F.softplus(bound)
    cover = soft_indicator(b - error, temp)
    c = (cover * weights).sum()

    return (c - p).pow(2)


def bound_loss_rank(error, bound, weights):
    """
    Used to force correlation so frame with 
    larger error get higher bound
    """
    weights = weights.reshape(-1) > 0

    b = F.softplus(bound).reshape(-1)

    error = error.reshape(-1)[weights] 
    b = b[weights]

    p = torch.randperm(error.numel(), device=error.device)
    s = torch.sign(error - error[p])

    return F.softplus(-s * (b - b[p])).mean()
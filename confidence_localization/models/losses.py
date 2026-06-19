"""Confidence (negative-log-likelihood) losses for the DOA heads.

Moved out of the old ``model.py`` (which also held a pile of unused CNN/UNet
baselines). Selected per ``cfg.nnl_func`` in each model's ``__init__``.
"""

import torch


def gaussian_loss(mean_err_squared, log_std, log_var_weight=1):
    return (0.5 * (mean_err_squared * torch.exp(-log_std)) + (log_var_weight * log_std)).mean()


def von_mises_loss(mean_err_squared, log_std, log_var_weight=1):
    angle_error = torch.sqrt(mean_err_squared).clamp(0, torch.pi)
    log_kappa = -2 * log_std
    # since the float is numerically limited, a max kappa of 500 is good enough
    kappa = torch.exp(log_kappa).clamp(max=5e2).to(torch.float64)

    log_I0 = torch.log(torch.i0(kappa) + 1e-8)
    # von Mises NLL: -log(I0(kappa)) - kappa * cos(error)
    nll = -(kappa * torch.cos(angle_error) - log_I0) + log_var_weight * log_kappa
    return nll.mean()

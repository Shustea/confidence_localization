import pytorch_lightning as pl
import torch
from numpy import arange, unique, deg2rad, floor, deg2rad
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from model import *
import torch.nn.init as init
from torchvision import transforms
import matplotlib.pyplot as plt

from torch.special import i0e

import torch.utils.checkpoint as cp

from mamba_ssm import Mamba

import hydra

import os
import sys
sys.path.extend([
    os.path.join(os.getcwd(), p) for p in ['confidence_localization', 'data']
])

import confidence_localization_dataloader as cld
from util import save_sample_as_image


class MambaResChannel(nn.Module):
    def __init__(self, cfg, expand: int = 2):
        super().__init__()
        channels = (
            getattr(cfg, "receivers_num", None)
            or getattr(cfg, "recivers_num", None)
            or getattr(cfg, "channels", None)
            )
        channels = 2 * (channels - 1)

        d_model = getattr(cfg, "d_model", None)
        if channels is None and d_model is None:
            raise AttributeError("Provide cfg.receivers_num (or channels) and/or cfg.d_model.")
        if channels is None:
            channels = d_model
        if d_model is None:
            d_model = channels
        self.channels = channels
        self.d_model = d_model
        self.use_proj = (channels != d_model)
        if self.use_proj:
            self.in_proj  = nn.Linear(channels, d_model)
            self.out_proj = nn.Linear(d_model, channels)
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=getattr(cfg, "hidden_dim", 64),
            d_conv=getattr(cfg, "conv_dim", 4),
            expand=expand
        )
        self.dropout = nn.Dropout(getattr(cfg, "dropout", 0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, F, T, C = x.shape
        if C != self.channels:
            raise ValueError(f"Expected C={self.channels}, got {C}")
        L = F * T
        y = x.reshape(B, L, C)
        if self.use_proj:
            y = self.in_proj(y)
        y = self.norm(y)
        y = self.mamba(y)
        if self.use_proj:
            y = self.out_proj(y)
        y = self.dropout(y)
        y = y.reshape(B, F, T, C)
        return x + y

class MambaResFreq(nn.Module):
    def __init__(self, cfg, expand=2):
        super().__init__()
        self.mamba = Mamba(
            d_model=cfg.d_model,  # model dimension
            d_state=cfg.hidden_dim,  # state dim
            d_conv=cfg.conv_dim,  # conv dim
            expand=expand  # FFN expansion ratio
        )

    def forward(self, x):
        B, Freq, T, d = x.shape
        return (x + self.mamba(x.reshape(B * Freq, T, d)).reshape(B ,Freq, T, d))
    
class MambaResTime(nn.Module):
    def __init__(self, cfg, expand=2):
        super().__init__()
        self.mamba = Mamba(
            d_model=cfg.d_model,  # model dimension
            d_state=cfg.hidden_dim,  # state dim
            d_conv=cfg.conv_dim,  # conv dim
            expand=expand  # FFN expansion ratio
        )

    def forward(self, x):
        B, Freq, T, d = x.shape
        return (x + self.mamba(x.permute(0, 2, 1, -1).reshape(B * T, Freq, d)).reshape(B ,Freq, T, d))
    
class MambaResTF(nn.Module):
    def __init__(self, cfg, expand=2):
        super().__init__()
        self.mambaT = MambaResTime(cfg, expand)
        self.mambaF = MambaResFreq(cfg, expand)
        # self.layer_norm = nn.LayerNorm([cfg.freq_dim, cfg.time_dim, cfg.d_model])

    def forward(self, x):
        return self.mambaF(self.mambaT(x))
    
class MambaResCTF(nn.Module):
    def __init__(self, cfg, expand=2):
        super().__init__()
        self.mambaTF = MambaResTF(cfg, expand)
        self.mambaC = MambaResChannel(cfg, expand)

    def forward(self, x):
        return self.mambaTF(self.mambaC(x))

class DOAMAMBA(pl.LightningModule):
    def __init__(self, cfg):
        super(DOAMAMBA, self).__init__()

        self.cfg = cfg

        self.negative_log_likelihood_func = [gaussian_loss if cfg.nnl_func == 'gaussian' else von_mises_loss if cfg.nnl_func == 'von_mises' else None][0]
        
        self.mamba_layers = nn.Sequential(*[
            MambaResCTF(cfg, expand)
            for expand in cfg.layers
        ])

        self.hidden = nn.Linear(cfg.d_model, 1)

        # self.output_head = nn.Sequential(
        #     nn.Linear(cfg.input_dim, cfg.input_dim // 2),
        #     nn.GELU(),
        #     nn.LayerNorm(cfg.input_dim // 2),
        # )
        
        self.doa = nn.Sequential(
            nn.Linear(cfg.input_dim, 2),
            nn.Tanh()  # output in [-1, 1]
        )

        # self.log_std = nn.Linear(cfg.input_dim // 2, 1)

        
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.hidden.weight)
        init.zeros_(self.hidden.bias)

        # init.xavier_uniform_(self.doa.weight)
        # init.zeros_(self.doa.bias)

        # init.xavier_uniform_(self.log_std.weight)
        # init.zeros_(self.log_std.bias)

    def forward(self, x):
        x = x.permute(0, 2, -1, 1)
        
        for _, block in enumerate(self.mamba_layers):
            x = block(x.clamp(-10,10))

        x_hat = F.gelu(self.hidden(F.normalize(x,dim=-1)).squeeze(-1)).permute(0, 2, 1)  # (B, T, F)

        doa_vec = self.doa(x_hat)                                    

        # safe_log_std = torch.clamp(self.log_std(x_hat), min=-2.0, max=1.5)
        return F.normalize(doa_vec, dim=-1)
    
    def circ_error(self, angle):
        return ((angle + torch.pi) % (2 * torch.pi) - torch.pi).abs()

    def accuracy(self, est, gt, var=0.1):
        valid = ~torch.isnan(gt)
        if torch.numel(var) > 1:
            var = var[valid]
        return torch.sum(self.circ_error(est.unsqueeze(1)[valid] - gt[valid]) < var) / torch.sum(valid)
    
    def _sanitize(self, doa_vec, log_std):
        log_std = log_std.squeeze(-1)
        doa_vec = torch.nan_to_num(doa_vec, nan=0.0)
        log_std = torch.nan_to_num(log_std, nan=0.0, posinf=1.5, neginf=-4.0)
        return doa_vec, log_std

    def _labels_to_vec(self, labels):
        valid_mask = ~labels.isnan()
        labels_vec = torch.stack((labels.cos(), labels.sin()), dim=-1)
        return labels_vec, valid_mask

    def _angle_metric(self, u, v):
        d = u - v
        return 2.0 * torch.asin((d.norm(dim=-1).clamp(0.0, 2.0)) * 0.5)

    def _best_speaker(self, pred_unit, labels_vec, valid_mask):
        B, T, _ = labels_vec.shape
        pred_exp = pred_unit.unsqueeze(1).expand(-1, S, -1, -1)
        mean_ang = self._angle_metric(pred_exp, labels_vec).mean(-1)
        mean_ang[torch.isnan(mean_ang)] = float('inf')
        best_s = mean_ang.argmin(dim=1)
        best_s_expand = best_s.view(B, 1, 1)
        chosen_mask = valid_mask.gather(1, best_s_expand.expand(-1, 1, T)).squeeze(1)
        chosen_tgt  = labels_vec.gather(1, best_s_expand.unsqueeze(-1).expand(-1, 1, T, 2)).squeeze(1)
        return best_s, chosen_mask, chosen_tgt

    def _alpha(self):
        if self.current_epoch < self.cfg.initial_warmup:
            return 0.0
        denom = max(1, self.cfg.warmup - self.cfg.initial_warmup)
        return float(min(1.0, (self.current_epoch - self.cfg.initial_warmup) / denom))
    
    def _temporal_regularization(self, doa_vec):
        return torch.linalg.norm(torch.diff(doa_vec, dim=-1))

    def _noise_regularizer(self, log_std, chosen_mask, device, dtype):
        noise_mask = ~chosen_mask
        noise_logstd = log_std[noise_mask]
        noise_logstd = noise_logstd[torch.isfinite(noise_logstd)]
        if noise_logstd.numel():
            target_min = getattr(self.cfg, "noise_logstd_min", 0.0)
            return F.relu(target_min - noise_logstd).mean()
        return torch.tensor(0.0, device=device, dtype=dtype)

    def loss(self, doa_vec, labels, batch_idx):
        B, T = labels.shape
        device, dtype = doa_vec.device, doa_vec.dtype
        # doa_vec, log_std = self._sanitize(doa_vec, log_std)
        labels_vec, valid_mask = self._labels_to_vec(labels)
        # best_s, chosen_mask, chosen_tgt = self._best_speaker(doa_vec, labels_vec, valid_mask)

        #flatten
        pred_flat, tgt_flat = doa_vec.reshape(B, -1, 2), labels_vec.reshape(B, -1, 2)

        # sel_logstd = logstd_flat[mask_flat]
        a = self._angle_metric(pred_flat, tgt_flat)
        a2 = a.clip(0, deg2rad(self.cfg.error_bound_on_mse)) * a.clip(0, deg2rad(self.cfg.error_bound_on_mse))
        # alpha = self._alpha()
        loss_main = a2.mean()
        # loss_nll = self.negative_log_likelihood_func(a2, sel_logstd, self.cfg.log_var_weight)
        # loss_main = (1 - alpha) * loss_ang + alpha * loss_nll
        mae = torch.abs(a).mean()

        # loss_noise = self._noise_regularizer(log_std, chosen_mask, device, dtype)
        # final_loss = loss_main + self.cfg.noise_beta * loss_noise + self.cfg.regularization_constant
        loss_main += self.cfg.temporal_reg_factor * self._temporal_regularization(pred_flat)
        return loss_main, mae.detach(), None
    
    def training_step(self, batch,  batch_idx):
        spectrum, labels, _ = batch

        doa = self(spectrum)
        
        train_loss, mae, _ = self.loss(doa, labels, batch_idx)

        with torch.autograd.set_detect_anomaly(True):
            train_loss.backward(retain_graph=True)

        self.log("train_loss", train_loss.mean(), on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, batch_size=self.cfg.batch_size)
        self.log("train_mae", mae.mean(), on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, batch_size=self.cfg.batch_size)

        return train_loss.to(dtype=torch.float32)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        spectrum, labels, title = batch
        B = spectrum.size(0)

        doa_unit = self(spectrum)
        # log_std = log_std.squeeze(-1)
        # std     = log_std.exp()
        doa     = torch.atan2(doa_unit[..., 1], doa_unit[..., 0])

        if batch_idx == 0 and labels.size(0) > 1:
            save_sample_as_image(doa[1].cpu(), labels[1].cpu(), 0.01*torch.ones(doa[1].shape).cpu(), title=f'{labels[1][0]}->{labels[1][-1]}', filename="DOA_1_example.png")
            # if ~torch.isnan(labels[1, 1].cpu()).any():
            #     save_sample_as_image(doa[1].cpu(), labels[1, 1].cpu(), 0.01*torch.ones(doa[1].shape).cpu(), title=f'{labels[1, 1][0]}->{labels[1, 1][-1]}', filename="DOA_2_example.png")

        val_loss, mae, best_spk = self.loss(doa_unit, labels, batch_idx)

        ref = labels[torch.arange(B, device=labels.device), best_spk]
        ang10  = self.accuracy(doa, ref, torch.tensor(deg2rad(10.0)))
        ang15  = self.accuracy(doa, ref, torch.tensor(deg2rad(15.0)))
        angStd = self.accuracy(doa, ref, torch.tensor(deg2rad(5.0)))

        # spk_mask  = ~ref.isnan()
        # std_spk   = std[spk_mask].mean()
        # std_noise = std[~spk_mask].mean()

        self.log_dict(
            {
                "validation_loss":         val_loss.mean(),
                "validation_mae":          torch.rad2deg(mae).mean(),
                "validation_accuracy_10":  ang10.mean(),
                "validation_accuracy_15":  ang15.mean(),
                "validation_accuracy_std": angStd.mean(),
                # "mean_std_over_speakers":  std_spk,
                # "mean_std_over_noise":     std_noise,
                "mean_MAE_over_speakers":  mae.mean(),
            },
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.cfg.batch_size,
        )

        return {"val_loss": val_loss.mean(), "val_acc": ang10.mean()}

    def configure_optimizers(self):
        # Use Adam optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        scheduler = {
        'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min'),
        'monitor': 'train_loss'
        }
        return {'optimizer': optimizer, 'scheduler': scheduler}
        

@hydra.main(config_path="..", config_name="config", version_base="1.1")
def main(cfg):

    # our_transform = transforms.Normalize(mean=[1/2, 1/2, 1/2, 1/2, 1/2, 1/2], std=[1/2, 1/2, 1/2, 1/2, 1/2, 1/2])
    train_loader = cld.get_dataloader(cfg, cfg.train_path)
    val_loader = cld.get_dataloader(cfg, cfg.val_path)

    logger = TensorBoardLogger("/workspaces/confidence_localization/logs", name="DOAMAMBA")

    model = DOAMAMBA(cfg)

    checkpoint_loss_callback = ModelCheckpoint(
    monitor="validation_loss",  # Monitor validation loss
    dirpath="./models/",  # Directory where the model is saved
    filename="best-loss-checkpoint-{epoch:02d}-{validation_loss_epoch:.2f}",
    save_top_k=2,  # Save only the best model
    mode="min",  # "min" for loss, "max" for accuracy/metrics
    save_last=True  # Save the last checkpoint
    )

    checkpoint_acc_callback_10 = ModelCheckpoint(
        monitor="validation_accuracy_10",
        dirpath="./models/",
        filename="best-acc10-{epoch:02d}-{validation_accuracy_10:.2f}",
        save_top_k=2,
        mode="max"
    )

    checkpoint_acc_callback_std = ModelCheckpoint(
        monitor="validation_accuracy_std",
        dirpath="./models/",
        filename="best-accstd-{epoch:02d}-{validation_accuracy_std:.2f}",
        save_top_k=2,
        mode="max"
    )

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=cfg.epochs,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",  
        devices=[cfg.default_gpu] if torch.cuda.is_available() else 0,
        gradient_clip_val=0.5,
        gradient_clip_algorithm="norm",
        # strategy='ddp',
        # sync_batchnorm=True,
        callbacks=[checkpoint_loss_callback, checkpoint_acc_callback_10, checkpoint_acc_callback_std]
    )
    ckpt_path = cfg.resume_from_checkpoint if ('resume_from_checkpoint' in cfg.keys()) else None
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()


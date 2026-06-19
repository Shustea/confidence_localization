"""Shared LightningModule base for every DOA model.

Holds all the model-agnostic machinery — loss, metrics, training/validation
steps, optimizer — that used to live inside ``DOAMAMBA``. Each concrete model
(DOAMAMBA, GccMamba, RtfMamba, ...) subclasses ``DOABase`` and only has to
define ``__init__`` (its layers) and ``forward`` returning
``(doa_unit[B,T,2], log_std[B,T,1])``.
"""

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from numpy import deg2rad

from util import save_sample_as_image, save_room_geometry


class DOABase(pl.LightningModule):
    """Base class: every method here only touches ``doa_vec`` / ``log_std`` /
    labels / vad, so it is independent of how the features are produced."""

    def circ_error(self, angle):
        return ((angle + torch.pi) % (2 * torch.pi) - torch.pi).abs()

    def accuracy(self, est, gt, var=0.1):
        valid = ~torch.isnan(gt)
        if torch.numel(var) > 1:
            var = var[valid]
        return torch.sum(
            self.circ_error(est.unsqueeze(1)[valid] - gt[valid]) < var
        ) / torch.sum(valid)

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
        mean_ang[torch.isnan(mean_ang)] = float("inf")
        best_s = mean_ang.argmin(dim=1)
        best_s_expand = best_s.view(B, 1, 1)
        chosen_mask = valid_mask.gather(1, best_s_expand.expand(-1, 1, T)).squeeze(1)
        chosen_tgt = labels_vec.gather(
            1, best_s_expand.unsqueeze(-1).expand(-1, 1, T, 2)
        ).squeeze(1)
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

    def loss(self, doa_vec, log_std, labels, batch_idx, vad=None):
        pred = F.normalize(doa_vec, dim=-1)

        if vad is None:
            w = torch.isfinite(labels).float()
        else:
            w = vad.float() * torch.isfinite(labels).float()

        # NaN-safe: zero out invalid labels so cos/sin/per don't propagate NaN
        # through the loss (RealMAN samples carry NaN labels during unvoiced gaps).
        safe_labels = torch.where(w > 0, labels, torch.zeros_like(labels))
        tgt = torch.stack((safe_labels.cos(), safe_labels.sin()), dim=-1)

        cos = (pred * tgt).sum(dim=-1).clamp(-1.0, 1.0)
        per = 1.0 - cos

        denom = w.sum().clamp_min(1.0)
        loss_main = (per * w).sum() / denom

        mae = torch.acos(cos.clamp(-0.999999, 0.999999))
        mae = (mae * w).sum().detach() / denom

        lam = float(getattr(self.cfg, "temporal_reg_factor", 0.0))
        if lam > 0 and pred.shape[1] > 1:
            w2 = w[:, 1:] * w[:, :-1]
            denom2 = w2.sum().clamp_min(1.0)
            tv = (pred[:, 1:] - pred[:, :-1]).norm(dim=-1)
            loss_main = loss_main + lam * (tv * w2).sum() / denom2

        return loss_main, mae, None

    def training_step(self, batch, batch_idx):
        spectrum, labels, _, vad, _wav = batch

        doa, log_std = self(spectrum)

        train_loss, mae, _ = self.loss(doa, log_std, labels, batch_idx, vad=vad)

        self.log(
            "train_loss",
            train_loss.mean(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.cfg.batch_size,
        )
        self.log(
            "train_mae",
            mae.mean(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.cfg.batch_size,
        )

        return train_loss.to(dtype=torch.float32)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        spectrum, labels, title, vad, wav = batch
        B = spectrum.size(0)

        doa_unit, log_std = self(spectrum)

        doa = torch.atan2(doa_unit[..., 1], doa_unit[..., 0])

        if batch_idx == 0 and labels.size(0) > 1:
            bound = log_std[1].squeeze(-1).exp()
            example_wav = wav[1].cpu()
            # Drop the sentinel zero-tensor that non-synthetic datasets emit.
            if example_wav.numel() <= 1:
                example_wav = None
            save_sample_as_image(
                doa[1].cpu(),
                labels[1].cpu(),
                bound.cpu(),
                filename="DOA_1_example.png",
                spectrum=spectrum[1].cpu(),
                waveform=example_wav,
                fs=int(self.cfg.fs),
                mic_positions=self.cfg.receivers_coords,
                source_radius=getattr(self.cfg, "source_radius", 1.825),
                room_dim=self.cfg.room_dim,
            )
            save_room_geometry(
                labels=labels[1].cpu(),
                room_dim=self.cfg.room_dim,
                mic_positions=self.cfg.receivers_coords,
                source_radius=getattr(self.cfg, "source_radius", 1.825),
                src_height=self.cfg.src_height,
                filename="room_geometry_example.png",
            )

        val_loss, mae, best_spk = self.loss(
            doa_unit, log_std, labels, batch_idx, vad=vad
        )

        # Per-frame wrapped angular error (rad). NaN labels get a sentinel 0 so the
        # arithmetic is finite; masks below exclude them from the aggregates.
        safe_lbl = torch.where(torch.isfinite(labels), labels, torch.zeros_like(labels))
        err_rad = self.circ_error(doa - safe_lbl)  # [B, T] in [0, π]
        err_sq_deg = (torch.rad2deg(err_rad)) ** 2

        mask_all = torch.isfinite(labels).float()  # only label-validity
        if vad is not None:
            mask_vad = mask_all * vad.float()  # AND VAD active
        else:
            mask_vad = mask_all
        n_all = mask_all.sum().clamp_min(1.0)
        n_vad = mask_vad.sum().clamp_min(1.0)

        # MAE
        mae_all_deg = (torch.rad2deg(err_rad) * mask_all).sum() / n_all
        mae_vad_deg = (torch.rad2deg(err_rad) * mask_vad).sum() / n_vad

        # RMS (degrees). Computed per frame, averaged with the per-mode mask.
        rmse_all_deg = ((err_sq_deg * mask_all).sum() / n_all).sqrt()
        rmse_vad_deg = ((err_sq_deg * mask_vad).sum() / n_vad).sqrt()

        # Accuracy @ angular threshold
        thr10 = torch.tensor(deg2rad(10.0), device=err_rad.device)
        thr15 = torch.tensor(deg2rad(15.0), device=err_rad.device)
        ang10_all = ((err_rad < thr10).float() * mask_all).sum() / n_all
        ang10_vad = ((err_rad < thr10).float() * mask_vad).sum() / n_vad
        ang15_all = ((err_rad < thr15).float() * mask_all).sum() / n_all
        ang15_vad = ((err_rad < thr15).float() * mask_vad).sum() / n_vad

        # MAE over inliers only (frames with per-frame error <= 15°). Complements
        # mae_vad which is dominated by catastrophic mispredictions on real data.
        inlier_mask_15 = mask_vad * (err_rad < thr15).float()
        n_inlier_15 = inlier_mask_15.sum().clamp_min(1.0)
        mae_vad_inlier15_deg = (torch.rad2deg(err_rad) * inlier_mask_15).sum() / n_inlier_15

        # Predicted-bound accuracy (coverage at the model's own ±σ band)
        bound_rad = log_std.squeeze(-1).exp()  # [B, T]
        ang_std_all = ((err_rad < bound_rad).float() * mask_all).sum() / n_all
        ang_std_vad = ((err_rad < bound_rad).float() * mask_vad).sum() / n_vad

        self.log_dict(
            {
                "validation_loss": val_loss.mean(),
                # MAE
                "validation_mae_all": mae_all_deg,
                "validation_mae_vad": mae_vad_deg,
                "validation_mae_vad_inlier15": mae_vad_inlier15_deg,
                # RMS angular error (deg)
                "validation_rmse_all": rmse_all_deg,
                "validation_rmse_vad": rmse_vad_deg,
                # Threshold accuracies
                "validation_accuracy_10_all": ang10_all,
                "validation_accuracy_10_vad": ang10_vad,
                "validation_accuracy_15_all": ang15_all,
                "validation_accuracy_15_vad": ang15_vad,
                # Coverage at the predicted bound
                "validation_accuracy_std_all": ang_std_all,
                "validation_accuracy_std_vad": ang_std_vad,
                # Fraction of frames the VAD considered active
                "validation_vad_active_frac": n_vad / n_all,
            },
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.cfg.batch_size,
        )

        return {"val_loss": val_loss.mean(), "val_acc": ang10_all}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min"
            ),
            "monitor": "train_loss",
        }
        return {"optimizer": optimizer, "scheduler": scheduler}

    def on_load_checkpoint(self, ckpt):
        ckpt.pop("optimizer_states", None)
        ckpt.pop("lr_schedulers", None)

import warnings
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.*deprecated.*")

from runtime_setup import configure_runtime

configure_runtime(__file__)

import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.checkpoint as cp
from einops import rearrange
from mamba_ssm import Mamba
from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
from numpy import deg2rad
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn

import confidence_localization_dataloader as cld
from model import ang_err_from_unit, kappa_to_circ_std, vm_nll_calibrated
from util import save_sample_as_image


class CompatibleMamba(Mamba):
    """Use the fused Mamba kernels on CUDA and a reference PyTorch path elsewhere."""

    def _reference_step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert (
            hidden_states.shape[1] == 1
        ), "Only support decoding with 1 token at a time for now"

        xz = self.in_proj(hidden_states.squeeze(1))
        x, z = xz.chunk(2, dim=-1)

        conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
        conv_state[:, :, -1] = x
        x = torch.sum(
            conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
        )
        if self.conv1d.bias is not None:
            x = x + self.conv1d.bias
        x = self.act(x).to(dtype=dtype)

        x_db = self.x_proj(x)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.linear(dt, self.dt_proj.weight)
        A = -torch.exp(self.A_log.float())

        dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
        dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
        dB = torch.einsum("bd,bn->bdn", dt, B)
        ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
        y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
        y = y + self.D.to(dtype) * x
        y = y * self.act(z)

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def forward(self, hidden_states, inference_params=None):
        if hidden_states.device.type == "cuda":
            return super().forward(hidden_states, inference_params=inference_params)

        batch, seqlen, _ = hidden_states.shape
        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                out, _, _ = self._reference_step(hidden_states, conv_state, ssm_state)
                return out

        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())
        x, z = xz.chunk(2, dim=1)

        if conv_state is not None:
            conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))

        x = self.act(self.conv1d(x)[..., :seqlen])
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

        y = selective_scan_ref(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=z,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=ssm_state is not None,
        )
        if ssm_state is not None:
            y, last_state = y
            ssm_state.copy_(last_state)
        y = rearrange(y, "b d l -> b l d")
        return self.out_proj(y)


class MambaResFreq(nn.Module):
    def __init__(self, cfg, expand=2):
        """Initialize the frequency-wise residual Mamba block."""
        super().__init__()
        self.mamba = CompatibleMamba(
            d_model=cfg.d_model,
            d_state=cfg.hidden_dim,
            d_conv=cfg.conv_dim,
            expand=expand,
        )

    def forward(self, x):
        """Apply the Mamba block across the time axis independently for each frequency bin."""
        B, Freq, T, d = x.shape
        y = self.mamba(x.reshape(B * Freq, T, d)).reshape(B, Freq, T, d)
        return x + y


class MambaResTime(nn.Module):
    def __init__(self, cfg, expand=2):
        """Initialize the time-wise residual Mamba block."""
        super().__init__()
        self.mamba = CompatibleMamba(
            d_model=cfg.d_model,
            d_state=cfg.hidden_dim,
            d_conv=cfg.conv_dim,
            expand=expand,
        )

    def forward(self, x):
        """Apply the Mamba block across the frequency axis independently for each time step."""
        B, Freq, T, d = x.shape
        y = (
            self.mamba(x.transpose(1, 2).reshape(B * T, Freq, d))
            .reshape(B, T, Freq, d)
            .transpose(1, 2)
        )
        return x + y


class MambaResChannel(nn.Module):
    def __init__(self, cfg, expand: int = 2):
        """Initialize the channel-mixing residual Mamba block."""
        super().__init__()
        channels = (
            getattr(cfg, "receivers_num", None) or getattr(cfg, "recivers_num", None)
        ) - 1
        d_model = getattr(cfg, "d_model", channels)
        self.channels, self.d_model = channels, d_model
        self.use_proj = channels != d_model
        if self.use_proj:
            self.in_proj = nn.Linear(channels, d_model)
            self.out_proj = nn.Linear(d_model, channels)
        self.norm = nn.LayerNorm(d_model)
        self.mamba = CompatibleMamba(
            d_model=d_model,
            d_state=getattr(cfg, "hidden_dim", 64),
            d_conv=getattr(cfg, "conv_dim", 4),
            expand=expand,
        )
        self.dropout = nn.Dropout(getattr(cfg, "dropout", 0.0))

    def forward(self, x):
        """Project channel features if needed, run the Mamba mixer, and add the residual back."""
        B, F, T, C = x.shape
        y = x.reshape(B, F * T, C)
        if self.use_proj:
            y = self.in_proj(y)
        y = self.mamba(self.norm(y))
        if self.use_proj:
            y = self.out_proj(y)
        y = self.dropout(y).reshape(B, F, T, C)
        return x + y


class MambaResTF(nn.Module):
    def __init__(self, cfg, expand=2):
        """Initialize the combined time-frequency residual block."""
        super().__init__()
        self.mambaT = MambaResTime(cfg, expand)
        self.mambaF = MambaResFreq(cfg, expand)
        # self.layer_norm = nn.LayerNorm([cfg.freq_dim, cfg.time_dim, cfg.d_model])

    def forward(self, x):
        """Apply the time and frequency residual mixers in sequence."""
        return self.mambaF(self.mambaT(x))


class MambaResCTF(nn.Module):
    def __init__(self, cfg, expand=2):
        """Initialize the composite channel-time-frequency residual block."""
        super().__init__()
        self.mambaTF = MambaResTF(cfg, expand)
        self.mambaC = MambaResChannel(cfg, expand)

    def forward(self, x):
        """Apply time-frequency mixing followed by channel mixing."""
        return self.mambaC(self.mambaTF(x))


class DOAMAMBA(pl.LightningModule):
    def __init__(self, cfg):
        """Initialize the DOA model, confidence head, and stacked residual Mamba backbone."""
        super(DOAMAMBA, self).__init__()

        self.cfg = cfg
        self.strict_loading = False

        self.alpha = torch.nn.Parameter(torch.ones(()))

        # self.negative_log_likelihood_func = [gaussian_loss if cfg.nnl_func == 'gaussian' else von_mises_loss if cfg.nnl_func == 'von_mises' else None][0]

        self.mamba_layers = nn.Sequential(
            *[MambaResCTF(cfg, expand) for expand in cfg.layers]
        )

        self.hidden = nn.Linear(cfg.d_model, 1)

        self.doa = nn.Sequential(
            nn.Linear(cfg.input_dim, 2), nn.Tanh()  # output in [-1, 1]
        )

        self.hidden_bound = nn.Linear(cfg.d_model, 1)
        self.kappa = nn.Linear(cfg.input_dim, 1)

        self.reset_parameters()

    def reset_parameters(self):
        """Reset the learnable heads with Xavier initialization and zero biases."""
        init.xavier_uniform_(self.hidden.weight)
        init.zeros_(self.hidden.bias)

        init.xavier_uniform_(self.hidden_bound.weight)
        init.zeros_(self.hidden_bound.bias)

        init.xavier_uniform_(self.kappa.weight)
        init.zeros_(self.kappa.bias)

    def forward(self, x):
        """Run the DOA model on an RTF batch and return a unit-vector direction estimate plus concentration.

        Example:
            Input: ``x`` shaped like ``[B, M - 1, T, F]``.
            Output: ``doa_unit`` with shape ``[B, T, 2]`` and ``kappa`` with
            shape ``[B, T]``.
        """
        x = x.permute(0, -1, 2, 1).contiguous()

        use_ckpt = self.training and getattr(self.cfg, "gradient_checkpointing", False)
        for block in self.mamba_layers:
            x_norm = F.normalize(x, dim=-1)
            x = (
                cp.checkpoint(block, x_norm, use_reentrant=False)
                if use_ckpt
                else block(x_norm)
            )

        x_norm = F.normalize(x, dim=-1)
        x_hat = F.gelu(self.hidden(x_norm).squeeze(-1)).permute(0, 2, 1)
        x_bound_hat = torch.tanh(self.hidden_bound(x_norm).squeeze(-1)).permute(0, 2, 1)

        doa_vec = self.doa(x_hat)
        doa_unit = F.normalize(doa_vec, dim=-1)

        kappa = F.softplus(self.kappa(x_bound_hat).squeeze(-1)) + float(
            getattr(self.cfg, "bound_min", 1e-3)
        )
        return doa_unit, kappa

    def circ_error(self, angle):
        """Compute absolute wrapped angular error."""
        return ((angle + torch.pi) % (2 * torch.pi) - torch.pi).abs()

    def accuracy(self, est, gt, std=0.1):
        """Measure the fraction of valid frames whose angular error stays below the provided tolerance."""
        valid = ~torch.isnan(gt)
        if torch.numel(std) > 1:
            std = std[valid]
        return torch.mean((self.circ_error(est[valid] - gt[valid]) < std).float())

    def loss(self, doa_vec, kappa, labels, vad=None):
        """Compute the training objective and summary metrics for a batch of DOA predictions.

        Input: ``doa_vec`` [B, T, 2], ``kappa`` [B, T], ``labels`` [B, T].
        Output: ``(total_loss, mae)`` — calibrated von Mises NLL and mean angular error.
        """
        pred = F.normalize(doa_vec, dim=-1)

        w = torch.isfinite(labels).float()
        if vad is not None:
            w = w * vad.float()

        cos = (
            (pred * torch.stack((labels.cos(), labels.sin()), dim=-1))
            .sum(dim=-1)
            .clamp(-1.0, 1.0)
        )
        mae = torch.acos(cos.clamp(-0.999999, 0.999999)) * w
        denom = w.sum().clamp_min(1.0)

        lam = float(getattr(self.cfg, "temporal_reg_factor", 0.0))
        if lam > 0 and pred.shape[1] > 1:
            w2 = w[:, 1:] * w[:, :-1]
            tv = (pred[:, 1:] - pred[:, :-1]).norm(dim=-1)
            temporal_loss = lam * (tv * w2).sum() / w2.sum().clamp_min(1.0)
        else:
            temporal_loss = 0.0

        mode = str(getattr(self.cfg, "train_mode", "doa")).lower()
        if mode == "doa":
            total_loss = (mae[w > 0].sum() / denom) ** 2 + temporal_loss
        else:
            error = ang_err_from_unit(pred, labels)
            total_loss = vm_nll_calibrated(error, kappa, self.alpha) + temporal_loss

        return total_loss, mae[w > 0].sum().detach() / denom

    def training_step(self, batch, batch_idx):
        """Run one Lightning training step and log the main optimization metrics."""
        spectrum, labels, _ = batch
        doa, kappa = self(spectrum)
        train_loss, mae = self.loss(doa, kappa, labels)
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
        self.log(
            "cal_alpha",
            self.alpha,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.cfg.batch_size,
        )
        return train_loss.to(dtype=torch.float32)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        """Run one validation step, compute angular metrics, and log calibration behavior."""
        spectrum, labels, _ = batch

        doa_unit, kappa = self(spectrum)
        circ_std = kappa_to_circ_std(self.alpha.exp() * kappa)
        doa = torch.atan2(doa_unit[..., 1], doa_unit[..., 0])

        if batch_idx == 0 and labels.size(0) > 1:
            save_sample_as_image(
                doa[1].cpu(),
                labels[1].cpu(),
                circ_std[1].cpu(),
                title=f"{labels[1][0]}->{labels[1][-1]}",
                filename="DOA_1_example.png",
            )

        val_loss, mae = self.loss(doa_unit, kappa, labels)

        ref = labels
        ang10 = self.accuracy(doa, ref, torch.tensor(deg2rad(10.0)))
        ang15 = self.accuracy(doa, ref, torch.tensor(deg2rad(15.0)))
        angStd = self.accuracy(doa, ref, circ_std)
        angBound = self.accuracy(doa, ref, 1 / kappa.rsqrt())

        self.log_dict(
            {
                "validation_loss": val_loss.mean(),
                "validation_mae": torch.rad2deg(mae).mean(),
                "validation_accuracy_10": ang10.mean(),
                "validation_accuracy_15": ang15.mean(),
                "validation_accuracy_std": angStd.mean(),
                "validation_accuracy_std_err": torch.abs(angStd.mean() - 0.68),
                "validation_accuracy_bound": angBound.mean(),
                # "mean_std_over_noise":     std_noise
                "cal_alpha": self.alpha,
            },
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.cfg.batch_size,
        )

        return {"val_loss": val_loss.mean(), "val_acc": ang10.mean()}

    def configure_optimizers(self):
        """Create the optimizer and learning-rate scheduler used during training."""
        opt = torch.optim.AdamW(
            (p for p in self.parameters() if p.requires_grad),
            lr=self.cfg.lr,
            weight_decay=getattr(self.cfg, "weight_decay", 1e-5),
        )
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=getattr(self.cfg, "lr_factor", 0.5),
            patience=getattr(self.cfg, "lr_patience", 10),
            min_lr=getattr(self.cfg, "lr_min", 1e-6),
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": "validation_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_load_checkpoint(self, ckpt):
        """Drop optimizer state from loaded checkpoints so resumed fine-tuning can start cleanly."""
        ckpt.pop("optimizer_states", None)
        ckpt.pop("lr_schedulers", None)

    def train_logstd_only(self):
        """Freeze most parameters and leave the confidence-related heads trainable."""
        for p in self.parameters():
            p.requires_grad = False

        for p in self.kappa.parameters():
            p.requires_grad = True
        for p in self.hidden_bound.parameters():
            p.requires_grad = True
        # self.alpha.requires_grad = True


def _build_trainer(cfg, callbacks):
    return pl.Trainer(
        logger=TensorBoardLogger(cfg.log_dir, name="DOAMAMBA"),
        max_epochs=cfg.epochs,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=[cfg.default_gpu] if torch.cuda.is_available() else 0,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        gradient_clip_val=0.5,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=getattr(cfg, "accumulate_grad_batches", 1),
        callbacks=callbacks,
    )


def _load_weights(model, ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state.get("state_dict", state), strict=False)
    print(f"Loaded weights from '{ckpt_path}'.")


@hydra.main(config_path="..", config_name="config", version_base="1.1")
def main(cfg):
    """Launch training in one of two modes (set via ``train_mode`` in config.yaml):

    doa        — train the full model end-to-end; optionally resume from a checkpoint.
    confidence — load a trained DOA checkpoint, freeze the DOA head, and train the
                 confidence (kappa) head only.
    """
    mode = str(getattr(cfg, "train_mode", "doa")).lower()
    ckpt = getattr(cfg, "resume_from_checkpoint", None) or None

    train_loader = cld.get_dataloader(cfg, shuffle=True, stage="train")
    val_loader = cld.get_dataloader(cfg, stage="val")

    model = DOAMAMBA(cfg)
    if ckpt:
        _load_weights(model, ckpt)

    if mode == "confidence":
        if not ckpt:
            raise ValueError(
                "train_mode=confidence requires resume_from_checkpoint to be set."
            )
        model.train_logstd_only()
        print("Confidence mode: DOA head frozen, training bound head only.")
    else:
        print("DOA mode: training full model.")

    callbacks = [
        EarlyStopping(
            monitor="validation_loss",
            patience=getattr(cfg, "early_stopping_patience", 25),
            min_delta=getattr(cfg, "early_stopping_min_delta", 1e-4),
            mode="min",
            verbose=True,
        ),
        ModelCheckpoint(
            monitor="validation_loss",
            dirpath=cfg.save_dir,
            filename="best-loss-{epoch:02d}-{validation_loss:.4f}",
            save_top_k=2,
            mode="min",
            save_last=True,
        ),
        ModelCheckpoint(
            monitor="validation_accuracy_10",
            dirpath=cfg.save_dir,
            filename="best-acc10-epoch={epoch}-validation_accuracy_10={validation_accuracy_10:.2f}",
            save_top_k=2,
            mode="max",
        ),
        ModelCheckpoint(
            monitor="validation_accuracy_std_err",
            dirpath=cfg.save_dir,
            filename="best-calibration-{epoch:02d}-{validation_accuracy_std:.2f}",
            save_top_k=2,
            mode="min",
        ),
    ]

    _build_trainer(cfg, callbacks).fit(model, train_loader, val_loader)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()

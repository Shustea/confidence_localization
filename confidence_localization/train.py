import pytorch_lightning as pl
import torch
from numpy import arange, unique, deg2rad, floor
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from model import *
import torch.nn.init as init
from torchvision import transforms
import matplotlib.pyplot as plt


from mamba_ssm import Mamba

import hydra

import os
import sys
sys.path.append(os.getcwd() + '/confidence_localization')
sys.path.append(os.getcwd() + '/data')
sys.path.append(os.getcwd() + '/confidence_localization')

import confidence_localization_dataloader as cld
from util import save_sample_as_image, save_doas
from util import save_sample_as_image, save_doas

class MambaRes(nn.Module):
    def __init__(self, cfg, expand=2):
        super().__init__()
        self.mamba = Mamba(
            d_model=cfg.d_model,  # model dimension
            d_state=cfg.hidden_dim,  # state dim
            d_conv=cfg.conv_dim,  # conv dim
            expand=expand  # FFN expansion ratio
        )

    def forward(self, x):
        return x + self.mamba(x)

class DOAMAMBA(pl.LightningModule):
    def __init__(self, cfg):
        super(DOAMAMBA, self).__init__()

        self.cfg = cfg
        self.channel_encoder = ChannelCNN(cfg.recivers_num, cfg.d_model)
        
        self.mamba_layers = nn.Sequential(*[
            MambaRes(cfg, 2 ** (floor(index / 2) + 1))
            for index in range(cfg.num_layers)
        ])

        self.hidden = nn.Linear(cfg.d_model, cfg.input_dim)

        self.doa = nn.Linear(cfg.input_dim, 2)

        self.log_std = nn.Linear(cfg.input_dim, 1)
        
        self.reset_parameters()

    def reset_parameters(self):
    
        init.xavier_uniform_(self.hidden.weight)
        init.zeros_(self.hidden.bias)

        init.xavier_uniform_(self.doa.weight)
        init.zeros_(self.doa.bias)

        # init.xavier_uniform_(self.std.weight)
        # init.zeros_(self.std.bias)

    def forward(self, x):
        x = self.channel_encoder(x)
        B, Freq, T, d = x.shape
        
        x = self.mamba_layers(x.reshape(B * Freq, T, d)).reshape(B ,Freq, T, d)

        x_hat = F.gelu(self.hidden(x)).permute(0, 2, 1, -1)

        doa_unit_vector = F.gelu(self.doa(x_hat))
        doa_unit_vector = F.normalize(doa_unit_vector, dim=-1).squeeze()
        
        safe_log_std = torch.clamp(self.log_std(x_hat), min=-20.0, max=3.5)

        return doa_unit_vector, safe_log_std
    
    def circ_error(self, angle):
        return ((angle + torch.pi) % (2 * torch.pi) - torch.pi).abs()

    def accuracy(self, est, gt, var=0.1):
        valid = ~torch.isnan(gt)
        return torch.sum(self.circ_error(est[valid] - gt[valid]) < var) / torch.sum(valid)
    
    def loss(self, doa_vec, log_std, labels):
        log_std = log_std.squeeze()
        labels = labels.squeeze()
        doa_vec = doa_vec.squeeze()
        labels_vec = torch.stack([torch.cos(labels), torch.sin(labels)], dim=-1)
        valid_mask = ~torch.isnan(labels)  # (B, T, F)

        doa_vec = doa_vec[valid_mask, :]          # (N, 2)
        labels_vec = labels_vec[valid_mask, :]    # (N, 2)
        log_std = log_std[valid_mask]                  # (N,)

        cos_sim = F.cosine_similarity(doa_vec, labels_vec, dim=-1).clamp(-1 + 1e-6, 1 - 1e-6)
        err = torch.acos(cos_sim)

        loss = ((err ** 2) / log_std.exp() + log_std).mean()

        if self.cfg.logvar_confidence_on_noise:
            loss += torch.norm(torch.log(log_std ** 2)).mean()

        return loss, err.mean()

    def training_step(self, batch,  batch_idx):
        spectrum, labels, _ = batch

        doa, std = self(spectrum)
        
        train_loss, _ = self.loss(doa, std, labels)

        self.log("train_loss", train_loss.mean(), on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, batch_size=self.cfg.batch_size)

        return train_loss.to(dtype=torch.float32)

    def validation_step(self, batch, batch_idx):
        spectrum, labels, title = batch
        # spectrum = self.batch_norm(spectrum.float())
        doa_unit, log_std = self(spectrum)
        doa = torch.atan2(doa_unit[..., 1], doa_unit[..., 0])
        if batch_idx==1:
            if (len(unique(labels[1][~labels[1].isnan()].cpu())) > 1):
                save_sample_as_image(spectrum[0][1].permute(1, 0) / spectrum[0][1].max(), labels[1], title[1], 'example.png')
                save_sample_as_image(labels[1], labels[1], title[1], 'example_gt.png')
                save_sample_as_image(doa[1], labels[1], title[1],'DOA_example.png')
                save_sample_as_image(log_std[1].exp(), labels[1], title[1], 'std_example.png')
                save_doas(doa[1], title[1], 'doa_distribiution.png')
        val_loss, mae = self.loss(doa_unit, log_std, labels)
        acc = self.accuracy(doa, labels)
        
        self.log("validation_loss", val_loss.mean(), on_epoch=True, sync_dist=True, prog_bar=True, batch_size=self.cfg.batch_size)
        self.log("validation_accuracy", acc.mean(), on_epoch=True, sync_dist=True, prog_bar=True, batch_size=self.cfg.batch_size)
        self.log("mean_std_over_speakers", log_std.squeeze()[~labels.isnan()].exp().mean(), on_epoch=True, sync_dist=True, batch_size=self.cfg.batch_size)
        self.log("mean_std_over_noise", log_std.squeeze()[labels.isnan()].exp().mean(), on_epoch=True, sync_dist=True, batch_size=self.cfg.batch_size)
        self.log("mean_MAE_over_speakers", mae.mean(), on_epoch=True, sync_dist=True, batch_size=self.cfg.batch_size)

        return {"val_loss": val_loss.mean(), "val_acc": acc.mean()}

    def configure_optimizers(self):
        # Use Adam optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        scheduler = {
        'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min'),
        'monitor': 'train_loss'
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        

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

    checkpoint_acc_callback = ModelCheckpoint(
    monitor="validation_accuracy",  # Monitor validation acc
    dirpath="./models/",  # Directory where the model is saved
    filename="best-acc-checkpoint-{epoch:02d}-{validation_accuracy_epoch:.2f}",
    save_top_k=2,  # Save only the best model
    mode="max"  # "min" for loss, "max" for accuracy/metrics
    )


    trainer = pl.Trainer(
        logger=logger,
        max_epochs=cfg.epochs,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",  
        devices=[5] if torch.cuda.is_available() else 0,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        # strategy='ddp',
        # sync_batchnorm=True,
        callbacks=[checkpoint_loss_callback, checkpoint_acc_callback]
    )

    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()


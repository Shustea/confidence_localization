import pytorch_lightning as pl
import torch
from numpy import arange, unique, deg2rad
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

class DOAMAMBA(pl.LightningModule):
    def __init__(self, cfg):
        super(DOAMAMBA, self).__init__()

        self.cfg = cfg
        self.channel_encoder = ChannelCNN(cfg.recivers_num)

        self.mamba_layers = nn.Sequential(*[
            Mamba(
                d_model=cfg.input_dim,  # model dimension
                d_state=cfg.hidden_dim,  # state dim
                d_conv=cfg.conv_dim,  # conv dim
                expand=cfg.expand  # FFN expansion ratio
            )
            for _ in range(cfg.num_layers)
        ])

        self.hidden = nn.Linear(cfg.input_dim, cfg.input_dim)

        self.doa = nn.Linear(cfg.input_dim, cfg.input_dim)

        self.logvar = nn.Linear(cfg.input_dim, cfg.input_dim)
        
        self.reset_parameters()

    def reset_parameters(self):
    
        init.xavier_uniform_(self.hidden.weight)
        init.zeros_(self.hidden.bias)

        init.xavier_uniform_(self.doa.weight)
        init.zeros_(self.doa.bias)

        init.xavier_uniform_(self.logvar.weight)
        init.zeros_(self.logvar.bias)

    def forward(self, x):
        x = self.channel_encoder(x)

        x = (x - x.mean((0, 1))) / x.std((0,1))
        x = self.mamba_layers(x)

        x_hat = F.relu(self.hidden(x))

        doa = F.silu(self.doa(x_hat))
        logvar = F.silu(self.logvar(x_hat))

        return doa, logvar
    
    def circ_error(self, angle):
        return ((angle + torch.pi) % (2 * torch.pi) - torch.pi).abs()

    def accuracy(self, est, gt, var):
        return torch.sum(self.circ_error(est - gt) < var) / torch.numel(est)
    
    def loss(self, doa, logvar, labels):
        std = torch.clamp(logvar.exp(), min=1e-6, max=1e6)
        mae = self.circ_error(doa[~labels.isnan()] - labels[~labels.isnan()]).mean()
        loss = (1/(2 * std) * (mae ** 2) + logvar / 2).mean()

        if self.cfg.logvar_confidence_on_noise:
            logvar_loss = torch.zeros_like(loss)
            logvar_loss[~labels.isnan()] = logvar[~labels.isnan()]
            logvar_loss[~labels.isnan()] = (torch.pi - logvar[labels.isnan()]) # we might add temp in future

            loss+= logvar_loss / 2

        # ways to improve loss:
        #
        # instead of adding logvar add logvar[~labels.isnan()] + [labels.isnan()]
        #
        # maybe we should force the accuracy to be ~68% (std inclusion rate)
        # (self.accuracy(doa, labels, logvar.exp()) - 68.2).abs()
        return loss, mae

    def training_step(self, batch, batch_idx):
        spectrum, labels = batch

        doa, logvar = self(spectrum)
        
        train_loss, _ = self.loss(doa, logvar, labels)

        self.log("train_loss", train_loss.mean(), on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)

        return train_loss.to(dtype=torch.float32)

    def validation_step(self, batch, batch_idx):
        spectrum, labels = batch
        # spectrum = self.batch_norm(spectrum.float())
        doa, logvar = self(spectrum)
        if batch_idx==0:
            if (len(unique(labels[1][~labels[1].isnan()].cpu())) > 1):
                save_sample_as_image(spectrum[1], labels[1], 'example.png')
                save_sample_as_image(labels[1], labels[1], 'example_gt.png')
                save_sample_as_image(doa[1], labels[1],'DOA_example.png')
                save_sample_as_image(logvar[1].exp(), labels[1], 'logvar_example.png')
                save_doas(doa[1], labels[1], 'doa_distribiution.png')
        val_loss, mae = self.loss(doa, logvar, labels)
        acc = self.accuracy(doa, labels, deg2rad(1))
        
        self.log("validation_loss", val_loss.mean(), on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("validation_accuracy", acc.mean(), on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("mean_std_over_speakers", logvar.exp()[~labels.isnan()].mean(), on_epoch=True, sync_dist=True)
        self.log("mean_std_over_noise", logvar.exp()[labels.isnan()].mean(), on_epoch=True, sync_dist=True)
        self.log("mean_MAE_over_speakers", mae.mean(), on_epoch=True, sync_dist=True)

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
    train_loader = cld.get_dataloader(cfg, cfg.train_path, cfg.interim_train_path)
    val_loader = cld.get_dataloader(cfg, cfg.val_path, cfg.interim_val_path)

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
        devices=[3] if torch.cuda.is_available() else 0,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        # strategy='ddp',
        # sync_batchnorm=True,
        callbacks=[checkpoint_loss_callback, checkpoint_acc_callback]
    )

    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()


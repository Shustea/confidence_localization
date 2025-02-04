import pytorch_lightning as pl
import torch
from numpy import arange
from torch import nn
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger
from model import Mamba
from torchvision import transforms

import hydra

import os
import sys
sys.path.append(os.getcwd() + '/data')

import confidence_localization_dataloader as cld

class DOAMAMBA(pl.LightningModule):
    def __init__(self, cfg):
        super(DOAMAMBA, self).__init__()
        self.cfg = cfg
        self.mamba_layers = nn.ModuleList([
            Mamba(cfg.input_dim, cfg.hidden_dim, cfg.recivers_num, cfg.selective_scan_flag)
            for _ in range(cfg.num_layers)
        ])
        # self.channel_conv = nn.Conv2d(2*(cfg.recivers_num - 1), 1, 2*(cfg.recivers_num - 1)-1, padding=2)
        self.hidden = nn.Linear(cfg.input_dim, cfg.input_dim)
        self.doa = nn.Linear(cfg.input_dim, cfg.input_dim)
        self.logvar = nn.Linear(cfg.input_dim, cfg.input_dim)
        # self.batch_norm = nn.BatchNorm2d(2*(cfg.recivers_num - 1))

    def forward(self, x):
        # feature extraction
        for layer in self.mamba_layers:
            x = F.tanh(layer(x))
        hidden = F.relu(self.hidden(x))
        return (self.doa(hidden), self.logvar(hidden))
    
    def loss(self, doa, logvar, labels):
        l1_loss = ((1/logvar.exp()) * F.l1_loss(doa[~labels.isnan()], labels[~labels.isnan()]) + logvar)
        # ways to improve loss:
        #
        # instead of adding logvar add logvar[~labels.isnan()] + [labels.isnan()]
        #
        # maybe we should force the accuracy to be ~68% (std inclusion rate)
        # (self.accuracy(doa, labels, logvar.exp()) - 68.2).abs()
        return l1_loss

    def training_step(self, batch, batch_idx):
        spectrum, labels = batch
        # spectrum = self.batch_norm(spectrum.float())
        doa, logvar = self(spectrum)
        #about loss - maybe the logvar should be more aggresive?
        train_loss = self.loss(doa, logvar, labels).mean()
        self.log("train_loss", train_loss, on_step=True, on_epoch=True, sync_dist=True)
        return train_loss.to(dtype=torch.float32)

    def validation_step(self, batch, batch_idx):
        spectrum, labels = batch
        # spectrum = self.batch_norm(spectrum.float())
        with torch.no_grad():
            doa, logvar = self(spectrum)
        mae = self.loss(doa, logvar, labels).mean()
        val_loss = ((1/logvar.exp()) * mae + logvar).mean()
        acc = self.accuracy(doa, labels, logvar.exp())
        
        self.log("validation_loss", val_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("validation_accuracy", acc, on_step=True, on_epoch=True, sync_dist=True)
        self.log("mean_std_over_speakers", logvar.exp()[~labels.isnan()].mean(), on_step=True, on_epoch=True, sync_dist=True)
        self.log("mean_std_over_noise", logvar.exp()[labels.isnan()].mean(), on_step=True, on_epoch=True, sync_dist=True)
        self.log("mean_MAE_over_speakers", mae.mean(), on_step=True, on_epoch=True, sync_dist=True)

        return {"val_loss": val_loss, "val_acc": acc}
    
    def accuracy(self, est, gt, var):
        return torch.sum(torch.abs(est - gt) < var.sqrt()) / (torch.numel(est))


    def configure_optimizers(self):
        # Use Adam optimizer
        return torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)


@hydra.main(config_path="..", config_name="config", version_base="1.1")
def main(cfg):
    train_loader = cld.get_dataloader(cfg, cfg.train_path)
    val_loader = cld.get_dataloader(cfg, cfg.val_path)

    logger = TensorBoardLogger("/workspaces/confidence_localization/logs", name="DOAMAMBA")

    model = DOAMAMBA(cfg)

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=50,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",  
        devices=[1,2] if torch.cuda.is_available() else 0,
        sync_batchnorm=True
    )

    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()


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
        self.mamba = Mamba(cfg.input_dim, cfg.hidden_dim, cfg.num_layers, cfg.recivers_num)
        self.channel_conv = nn.Conv2d(2*(cfg.recivers_num - 1), 1, 2*(cfg.recivers_num - 1)-1, padding=2)
        self.doa = nn.Linear(cfg.hidden_dim, cfg.input_dim)
        self.logvar = nn.Linear(cfg.hidden_dim, cfg.input_dim)
        self.batch_norm = nn.BatchNorm2d(2*(cfg.recivers_num - 1))

    def forward(self, x):
        # feature extraction
        embeded_space = F.selu(self.channel_conv(F.tanh(self.mamba(x))).squeeze(1))
        return (self.doa(embeded_space), self.logvar(embeded_space))

    def training_step(self, batch, batch_idx):

        loss_func = nn.MSELoss()
        spectrum, labels = batch
        spectrum = self.batch_norm(spectrum.float())
        doa, logvar = self(spectrum)
        #about loss - maybe the logvar should be more aggresive?
        train_loss = ((1/logvar.exp()) * torch.sqrt(loss_func(doa[~labels.isnan()], labels[~labels.isnan()])) + logvar).mean()
        self.log("train_loss", train_loss, on_step=True, on_epoch=True)
        return train_loss.to(dtype=torch.float32)

    def validation_step(self, batch, batch_idx):
        loss_func = nn.MSELoss()
        spectrum, labels = batch
        spectrum = self.batch_norm(spectrum.float())
        doa, logvar = self(spectrum)
        val_loss = ((1/logvar.exp()) * torch.sqrt(loss_func(doa[~labels.isnan()], labels[~labels.isnan()])) + logvar).mean()
        acc = self.accuracy(doa, labels, logvar.exp())
        
        self.log("validation_loss", val_loss, on_step=True, on_epoch=True)
        self.log("validation_accuracy", acc, on_step=True, on_epoch=True)

        return {"val_loss": val_loss, "val_acc": acc}
    
    def accuracy(self, est, gt, var):
        return torch.sum(torch.abs(est - gt) < var) / (torch.numel(est))


    def configure_optimizers(self):
        # Use Adam optimizer
        return torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)


@hydra.main(config_path="..", config_name="config", version_base="1.1")
def main(cfg):
    train_loader = cld.get_dataloader(cfg, cfg.train_path, transform=transforms.Normalize((0, 0, 0, 0, 0, 0),
                                                                                           (0.5, 0.5, 0.5, 0.5, 0.5, 0.5)))
    val_loader = cld.get_dataloader(cfg, cfg.val_path, transform=transforms.Normalize((0, 0, 0, 0, 0, 0),
                                                                                       (0.5, 0.5, 0.5, 0.5, 0.5, 0.5)))

    logger = TensorBoardLogger("/workspaces/confidence_localization/logs", name="DOAMAMBA")

    model = DOAMAMBA(cfg)

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=50,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",  
        devices=[2,3] if torch.cuda.is_available() else 0
    )

    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()


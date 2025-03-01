import pytorch_lightning as pl
import torch
from numpy import arange, unique
from torch import nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from model import ChannelCNN, ResidualBlock, RMSNorm, DOAMAMBA
from torchvision import transforms
import matplotlib.pyplot as plt

import hydra

import os
import sys
sys.path.append(os.getcwd() + '/data')

import confidence_localization_dataloader as cld

class DOAMAMBA(pl.LightningModule):
    def __init__(self, cfg):
        super(DOAMAMBA, self).__init__()
        self.cfg = cfg
        self.channel_conv = ChannelCNN(cfg.recivers_num)

        self.mamba_layers = nn.ModuleList([
            ResidualBlock(cfg)
            for _ in range(cfg.num_layers)
        ])

        self.hidden = nn.Linear(cfg.input_dim, cfg.input_dim)
        self.doa = nn.Linear(cfg.input_dim, cfg.input_dim)
        self.logvar = nn.Linear(cfg.input_dim, cfg.input_dim)

    def forward(self, x):
        x = self.channel_conv(x)
        for layer in self.mamba_layers:
            x = F.tanh(layer(x))
        x = F.relu(self.hidden(x))
        return (self.doa(x), self.logvar(x))
    
    def unwrap_angle(self, angle):
        return angle % (2 * torch.pi)

    def accuracy(self, est, gt, var):
        return torch.sum(torch.abs(est - gt) < var.sqrt()) / torch.numel(est)
    
    def loss(self, doa, logvar, labels):
        mae = F.l1_loss(doa[~labels.isnan()], labels[~labels.isnan()])
        l1_loss = (1/(logvar.exp()) * mae + logvar)
        # ways to improve loss:
        #
        # instead of adding logvar add logvar[~labels.isnan()] + [labels.isnan()]
        #
        # maybe we should force the accuracy to be ~68% (std inclusion rate)
        # (self.accuracy(doa, labels, logvar.exp()) - 68.2).abs()
        return l1_loss, mae

    def training_step(self, batch, batch_idx):
        spectrum, labels = batch
        # spectrum = self.batch_norm(spectrum.float())
        doa, logvar = self(spectrum)
        #about loss - maybe the logvar should be more aggresive?
        train_loss, _ = self.loss(doa, logvar, labels)
        self.log("train_loss", train_loss.mean(), on_step=True, on_epoch=True, sync_dist=True)
        return train_loss.mean().to(dtype=torch.float32)

    def validation_step(self, batch, batch_idx):
        spectrum, labels = batch
        # spectrum = self.batch_norm(spectrum.float())
        doa, logvar = self(spectrum)
        if batch_idx==0:
            if (len(unique(labels[1][~labels[1].isnan()].cpu())) > 1):
                save_sample_as_image(labels[1], labels[1], 'example_gt.png')
                save_sample_as_image(doa[1], labels[1],'DOA_example.png')
                save_sample_as_image(logvar[1], labels[1], 'logvar_example.png')
        val_loss, mae = self.loss(doa, logvar, labels)
        acc = self.accuracy(doa, labels, logvar.exp())
        
        self.log("validation_loss", val_loss.mean(), on_step=True, on_epoch=True, sync_dist=True)
        self.log("validation_accuracy", acc.mean(), on_step=True, on_epoch=True, sync_dist=True)
        self.log("mean_std_over_speakers", logvar.exp()[~labels.isnan()].mean(), on_step=True, on_epoch=True, sync_dist=True)
        self.log("mean_std_over_noise", logvar.exp()[labels.isnan()].mean(), on_step=True, on_epoch=True, sync_dist=True)
        self.log("mean_MAE_over_speakers", mae.mean(), on_step=True, on_epoch=True, sync_dist=True)

        return {"val_loss": val_loss.mean(), "val_acc": acc.mean()}

    def configure_optimizers(self):
        # Use Adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
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
    monitor="validation_loss_epoch",  # Monitor validation loss
    dirpath="./models/",  # Directory where the model is saved
    filename="best-loss-checkpoint-{epoch:02d}-{validation_loss_epoch:.2f}",
    save_top_k=2,  # Save only the best model
    mode="min",  # "min" for loss, "max" for accuracy/metrics
    save_last=True  # Save the last checkpoint
    )

    checkpoint_acc_callback = ModelCheckpoint(
    monitor="validation_accuracy_epoch",  # Monitor validation acc
    dirpath="./models/",  # Directory where the model is saved
    filename="best-acc-checkpoint-{epoch:02d}-{validation_accuracy_epoch:.2f}",
    save_top_k=2,  # Save only the best model
    mode="max",  # "min" for loss, "max" for accuracy/metrics
    save_last=True  # Save the last checkpoint
    )


    trainer = pl.Trainer(
        logger=logger,
        max_epochs=cfg.epochs,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",  
        devices=[5,6,7] if torch.cuda.is_available() else 0,
        strategy='ddp',
        sync_batchnorm=True,
        callbacks=[checkpoint_loss_callback, checkpoint_acc_callback]
    )

    trainer.fit(model, train_loader, val_loader)

def save_sample_as_image(tensor: torch.Tensor, label: torch.Tensor, filename: str, path='/workspaces/confidence_localization/samples/'):
    # Ensure tensor is on CPU and detach if it's a computation graph tensor
    if tensor.is_cuda:
        tensor = tensor.cpu()
    tensor = tensor.detach()

    plt.figure()
    plt.imshow(tensor.numpy().T, origin='lower')
    plt.axis("off")
    plt.colorbar()

    plt.title(f'{str(unique(label[~label.isnan()].cpu()))}', fontsize=14, fontweight="bold")

    # Save the image
    plt.savefig(path + filename, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()

if __name__ == "__main__":
    main()


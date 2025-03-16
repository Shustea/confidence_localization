import pytorch_lightning as pl
import torch
from numpy import arange, unique
from torch import nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from model import *
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
        self.channel_encoder = ChannelCNN(cfg.recivers_num)

        self.mamba_layers = nn.ModuleList([
            ResidualBlock(cfg)
            for _ in range(cfg.num_layers)
        ])

        self.doa = nn.Conv2d(1, cfg.num_classes, kernel_size=1)

        self.logvar = nn.Linear(cfg.num_classes, 1)

    def forward(self, x):
        x = self.channel_encoder(x).squeeze()

        for layer in self.mamba_layers:
            x = torch.tanh(layer(x))

        logits = self.doa(x.unsqueeze(1)).permute(0, -2, -1, 1)
        doa = F.softmax(logits, dim=-1) # Confidence distribution over classes.
        logvar = self.logvar(doa).squeeze(-1)  

        return logits, doa, logvar
    
    def unwrap_angle(self, angle):
        return angle % (2 * torch.pi)

    def accuracy(self, est, gt, var):
        return torch.sum(torch.abs(est - gt) < var.sqrt()) / torch.numel(est)
    
    def loss(self, logits, pred, logvar, labels):
        # Cross entropy loss: labels should be LongTensor with class indices.
        loss_val = F.cross_entropy(logits.to(dtype=float), labels.to(dtype=float))
        acc = torch.mean(torch.abs(self.accuracy(torch.argmax(pred, dim=-1),
                                        torch.argmax(labels, dim=-1), logvar.exp()) - 68.2))
        return loss_val + self.cfg.loss_gamma * acc

    def training_step(self, batch, batch_idx):
        spectrum, labels = batch
        logits, doa, logvar = self(spectrum)
        train_loss = self.loss(logits, doa, logvar, labels)
        self.log("train_loss", train_loss, on_step=True, on_epoch=True, sync_dist=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        spectrum, labels = batch
        logits, doa, logvar = self(spectrum)
        val_loss = self.loss(logits, doa, logvar, labels)

        # Calculate accuracy.
        preds = torch.argmax(doa, dim=-1)
        acc = (preds == torch.argmax(labels, dim=-1)).float().mean()

        self.log("validation_loss", val_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("validation_accuracy", acc, on_step=True, on_epoch=True, sync_dist=True)
        return {"val_loss": val_loss, "val_acc": acc}

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
        devices=[0,1,2] if torch.cuda.is_available() else 0,
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

def save_doas(tensor: torch.Tensor, label: torch.Tensor, filename: str, path='/workspaces/confidence_localization/samples/'):
    # Ensure tensor is on CPU and detach if it's a computation graph tensor
    if tensor.is_cuda:
        tensor = tensor.cpu()
    tensor = tensor.detach()

    plt.figure()
    plt.hist(tensor.numpy(), bins=20)

    plt.title(f'{str(unique(label[~label.isnan()].cpu()))}', fontsize=14, fontweight="bold")

    # Save the image
    plt.savefig(path + filename, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()

if __name__ == "__main__":
    main()


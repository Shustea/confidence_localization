import os
import sys
import torch, gc
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from numpy import arange
from model import Mamba
sys.path.append(os.getcwd() + '/data')
import confidence_localization_dataloader as cld
from torch.utils.tensorboard import SummaryWriter
import hydra
from omegaconf import DictConfig

class DOAMAMBA(nn.Module):
    def __init__(self, cfg):
        super(DOAMAMBA, self).__init__()
        self.cfg = cfg
        self.mamba_layers = nn.ModuleList([
            Mamba(cfg.input_dim, cfg.hidden_dim, cfg.recivers_num, cfg.selective_scan_flag)
            for _ in range(cfg.num_layers)
        ])
        self.hidden = nn.Linear(cfg.input_dim, cfg.input_dim)
        self.doa = nn.Linear(cfg.input_dim, cfg.input_dim)
        self.logvar = nn.Linear(cfg.input_dim, cfg.input_dim)

    def forward(self, x):
        for layer in self.mamba_layers:
            x = F.tanh(layer(x))
        hidden = F.relu(self.hidden(x))
        return (self.doa(hidden), self.logvar(hidden))

    def accuracy(self, est, gt, var):
        return torch.sum(torch.abs(est - gt) < var.sqrt()) / torch.numel(est)


def train(model, dataloader, optimizer, device, writer, epoch):
    model.train()
    total_loss = 0
    loss_func = nn.L1Loss()

    for batch_idx, batch in tqdm(enumerate(dataloader)):
        spectrum, labels = [x.to(device) for x in batch]

        optimizer.zero_grad()
        doa, logvar = model(spectrum)
        train_loss = (
            (1 / logvar.exp()) * loss_func(doa[~labels.isnan()], labels[~labels.isnan()])
            + logvar
        ).sum()
        train_loss.backward()
        optimizer.step()

        total_loss += train_loss.item()
        writer.add_scalar("Train/Loss", train_loss.item(), epoch * len(dataloader) + batch_idx)

    return total_loss / len(dataloader)


def validate(model, dataloader, device, writer, epoch):
    model.eval()
    total_loss = 0
    total_acc = 0
    loss_func = nn.L1Loss()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            spectrum, labels = [x.to(device) for x in batch]
            doa, logvar = model(spectrum)
            mae = loss_func(doa[~labels.isnan()], labels[~labels.isnan()])
            val_loss = ((1 / logvar.exp()) * mae + logvar).sum()
            acc = model.accuracy(doa[~labels.isnan()], labels[~labels.isnan()], logvar[~labels.isnan()].exp())

            total_loss += val_loss.item()
            total_acc += acc.item()

            global_step = epoch * len(dataloader) + batch_idx

            writer.add_scalar("Validation/Loss", val_loss.item(), global_step)
            writer.add_scalar("Validation/Accuracy", acc.item(), global_step)
            writer.add_scalar("Validation/Mean_std_over_speakers", logvar.exp()[~labels.isnan()].mean().item(), global_step)
            writer.add_scalar("Validation/Mean_std_over_noise", logvar.exp()[labels.isnan()].mean().item(), global_step)
            writer.add_scalar("Validation/Mean_MAE_over_speakers", mae.mean().item(), global_step)

    return total_loss / len(dataloader), total_acc / len(dataloader)

@hydra.main(config_path="..", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    # TensorBoard logger
    writer = SummaryWriter(log_dir=cfg.log_dir)
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    gc.collect()
    torch.cuda.empty_cache()

    # Prepare data loaders
    train_loader = cld.get_dataloader(cfg, cfg.train_path, num_workers=cfg.num_workers, shuffle=True)
    val_loader = cld.get_dataloader(cfg, cfg.val_path, num_workers=cfg.num_workers, shuffle=False)

    # Initialize model, optimizer, and device
    model = DOAMAMBA(cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Training loop
    for epoch in range(cfg.epochs):
        train_loss = train(model, train_loader, optimizer, device, writer, epoch)
        val_loss, val_acc = validate(model, val_loader, device, writer, epoch)

        print(f"Epoch {epoch + 1}/{cfg.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Save the model
    torch.save(model.state_dict(), os.path.join(cfg.save_dir, "doa_mamba_model.pth"))
    writer.close()

if __name__ == "__main__":
    main()

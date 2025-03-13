import pytorch_lightning as pl
import torch
from numpy import arange, unique
from torch import nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from model import *
from train import DOAMAMBA
from torchvision import transforms
import matplotlib.pyplot as plt

import hydra

import os
import sys
sys.path.append(os.getcwd() + '/data')

import confidence_localization_dataloader as cld

        

@hydra.main(config_path="..", config_name="config", version_base="1.1")
def main(cfg):
    # our_transform = transforms.Normalize(mean=[1/2, 1/2, 1/2, 1/2, 1/2, 1/2], std=[1/2, 1/2, 1/2, 1/2, 1/2, 1/2])
    val_loader = cld.get_dataloader(cfg, cfg.val_path, cfg.interim_val_path)

    model = DOAMAMBA(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = '/workspaces/unconditional_conditional_VAE/models/last-v4.ckpt'
    cl_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(cl_dict['state_dict'])
    model.eval()

    spectrum, labels = next(iter(val_loader))

    with torch.no_grad():
        doa, logvar = model.forward(spectrum)
        if (len(unique(labels[1][~labels[1].isnan()].cpu())) > 1):
                save_sample_as_image(labels[1], labels[1], 'test_gt.png')
                save_sample_as_image(doa[1], labels[1],'DOA_test.png')
                save_sample_as_image(logvar[1].exp(), labels[1], 'logvar_test.png')
                save_doas(doa[1], labels[1], 'test_doa_distribiution.png')

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


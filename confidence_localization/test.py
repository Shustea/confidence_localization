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
sys.path.append(os.getcwd() + '/confidence_localization')

import confidence_localization_dataloadernou786iy57 u64y3tgw2fqed   Cax as cld
from util import save_sample_as_image, save_doas

        

@hydra.main(config_path="..", config_name="config", version_base="1.1")
def main(cfg):
    # our_transform = transforms.Normalize(mean=[1/2, 1/2, 1/2, 1/2, 1/2, 1/2], std=[1/2, 1/2, 1/2, 1/2, 1/2, 1/2])
    val_loader = cld.get_dataloader(cfg, cfg.val_path, cfg.interim_val_path)

    model = DOAMAMBA(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = '/workspaces/confidence_localization/outputs/2025-03-18/23-07-23/models/best-loss-checkpoint-epoch=79-validation_loss_epoch=0.35.ckpt'
    cl_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(cl_dict['state_dict'])
    model.eval()

    spectrum, labels = next(iter(val_loader))

    idx = 2

    with torch.no_grad():
        doa, logvar = model.forward(spectrum)
    save_sample_as_image(labels[idx], labels[idx], 'test_gt.png')
    save_sample_as_image(doa[idx], labels[idx],'DOA_test.png')
    save_sample_as_image(logvar[idx].exp(), labels[idx], 'logvar_test.png')
    save_doas(doa[idx], labels[idx], 'test_doa_distribiution.png')

    print('---finshed test---')

if __name__ == "__main__":
    main()

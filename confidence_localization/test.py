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

import confidence_localization_dataloader as cld
from util import save_sample_as_image, save_doas

        

@hydra.main(config_path="..", config_name="config", version_base="1.1")
def main(cfg):
    # our_transform = transforms.Normalize(mean=[1/2, 1/2, 1/2, 1/2, 1/2, 1/2], std=[1/2, 1/2, 1/2, 1/2, 1/2, 1/2])
    val_loader = cld.get_dataloader(cfg, cfg.val_path)

    model = DOAMAMBA(cfg)

    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

    checkpoint_path = '/workspaces/confidence_localization/outputs/2026-01-18/21-35-46/models/best-acc10-epoch=116-validation_accuracy_10=0.91.ckpt'
    cl_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(cl_dict['state_dict'], strict=False)
    model.to(device).eval()

    spectrum, labels, title= next(iter(val_loader))
    spectrum = spectrum.to(device)
    labels = labels.to(device)

    idx = 1

    with torch.no_grad():
        doa_unit, log_std = model.forward(spectrum)
        
    log_std = log_std[idx].squeeze(-1)
    std     = log_std.exp()
    doa     = torch.atan2(doa_unit[..., 1], doa_unit[..., 0])
    doa = doa[idx]
    
    acc_map = model.circ_error(doa - labels[idx]) < std

    spk_idx = torch.argmax(acc_map.sum((1,2)))
    angle_diff = model.circ_error(doa - labels[idx, spk_idx])

    print(f' -  - - - - -- - - - -- - - -- -- - - - - -- - -')
    print(f'acc @ {(acc_map[spk_idx][~torch.isnan(acc_map[spk_idx])]).sum() / (~torch.isnan(acc_map[spk_idx])).sum()}')
    print(f'mean error @ {torch.rad2deg(angle_diff[~torch.isnan(angle_diff)].mean())}')
    print(f'20th quantile error @ {torch.rad2deg(angle_diff[~torch.isnan(angle_diff)].quantile(0.2))}')
    print(f'50th quantile error @ {torch.rad2deg(angle_diff[~torch.isnan(angle_diff)].quantile(0.5))}')
    print(f'70th quantile error @ {torch.rad2deg(angle_diff[~torch.isnan(angle_diff)].quantile(0.7))}')
    print(f'90th quantile error @ {torch.rad2deg(angle_diff[~torch.isnan(angle_diff)].quantile(0.9))}')
    print(f'95th quantile error @ {torch.rad2deg(angle_diff[~torch.isnan(angle_diff)].quantile(0.95))}')
    print(f' - --- - - -  - - -- - -- -- - - -- - -- - - - -')

    lbl_1 = labels[idx, spk_idx]
    if lbl_1.numel() and lbl_1[~lbl_1.isnan()].unique().numel() > 1:
        save_sample_as_image(doa.cpu(), lbl_1.cpu(), title[idx], "test_DOA.png")
        save_sample_as_image(std.cpu(), lbl_1.cpu(), title[idx], "test_std.png")
        save_sample_as_image(acc_map[spk_idx].cpu(), lbl_1.cpu(), title[idx], "test_acc.png")
        save_sample_as_image(acc_map[spk_idx].cpu(), lbl_1.cpu(), title[idx], "test_acc.png")
        save_sample_as_image(angle_diff.cpu(), lbl_1.cpu(), title[idx], "test_error.png")


    print('---finshed test---')

if __name__ == "__main__":
    main()

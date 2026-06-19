import os
import sys

import hydra
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

sys.path.extend(
    [os.path.join(os.getcwd(), p) for p in ["confidence_localization", "data"]]
)

import confidence_localization_dataloader as cld
from models import build_model
# Re-exported for backward compatibility with scripts importing them from train.
from models import DOAMAMBA, MambaResCTF  # noqa: F401


@hydra.main(config_path="..", config_name="config", version_base="1.1")
def main(cfg):

    # Snapshot the resolved config next to the checkpoints (./models/ under the
    # Hydra run dir) so the LOCATA benchmark can reconstruct this exact model
    # from just the checkpoint path. See locata_benchmark._discover_run_config.
    from omegaconf import OmegaConf
    os.makedirs("./models", exist_ok=True)
    OmegaConf.save(cfg, "./models/config.yaml")

    train_loader = cld.get_dataloader(cfg, stage="train", shuffle=True)
    val_loader = cld.get_dataloader(cfg, stage="val")

    logger = TensorBoardLogger(
        "/workspaces/confidence_localization/logs", name="DOAMAMBA"
    )

    model = build_model(cfg)

    ckpt = (
        cfg.resume_from_checkpoint if ("resume_from_checkpoint" in cfg.keys()) else None
    )
    if ckpt:
        model = type(model).load_from_checkpoint(ckpt, cfg=cfg, strict=False)

    checkpoint_loss_callback = ModelCheckpoint(
        monitor="validation_loss",  # Monitor validation loss
        dirpath="./models/",  # Directory where the model is saved
        filename="best-loss-checkpoint-{epoch:02d}-{validation_loss_epoch:.2f}",
        save_top_k=2,  # Save only the best model
        mode="min",  # "min" for loss, "max" for accuracy/metrics
        save_last=True,  # Save the last checkpoint
    )

    checkpoint_acc_callback_10 = ModelCheckpoint(
        monitor="validation_accuracy_10_vad",
        dirpath="./models/",
        filename="best-acc10-{epoch:02d}-{validation_accuracy_10_vad:.2f}",
        save_top_k=2,
        mode="max",
    )

    checkpoint_acc_callback_std = ModelCheckpoint(
        monitor="validation_accuracy_std_vad",
        dirpath="./models/",
        filename="best-accstd-{epoch:02d}-{validation_accuracy_std_vad:.2f}",
        save_top_k=2,
        mode="max",
    )

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=cfg.epochs,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=[cfg.default_gpu] if torch.cuda.is_available() else 0,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        gradient_clip_val=0.5,
        gradient_clip_algorithm="norm",
        callbacks=[
            checkpoint_loss_callback,
            checkpoint_acc_callback_10,
            checkpoint_acc_callback_std,
        ],
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()

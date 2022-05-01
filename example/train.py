#!/usr/bin/env python
import hydra
from hydra._internal import instantiate
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import Trainer
from meshgraphnet.callbacks import PlotsCallBack, WandbModelCheckpoint
from datetime import datetime
import wandb


def get_callbacks(config):
    monitor = {"monitor": "valid/loss_epoch", "mode": "min"}
    callbacks = [
        LearningRateMonitor(logging_interval='epoch'),
        WandbModelCheckpoint(
            save_last=True, save_top_k=1, every_n_epochs=1, filename="best", **monitor
        ),
    ]

    if config.early_stopping is not None:
        stopper = EarlyStopping(
            patience=int(config.early_stopping),
            min_delta=0,
            strict=False,
            check_on_train_epoch_end=False,
            **monitor,
        )
        callbacks.append(stopper)
    return callbacks


@hydra.main(config_path="../configs", config_name="train")
def train(cfg: DictConfig):
    print(cfg)
    wandb.init(
        entity=cfg.entity, project=cfg.project, group=cfg.group, resume="allow",
        name=datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    )
    pl.seed_everything(cfg.seed)
    datamodule = instantiate(cfg.dataloader)
    datamodule.prepare_data()
    datamodule.setup()

    # create model
    task = instantiate(cfg.task)
    wandb_logger = WandbLogger(project=cfg.project, group=cfg.group, config=cfg.__dict__, save_code=True)
    callbacks = get_callbacks(cfg)
    plots_callback = PlotsCallBack(datamodule.valid_ds,
                                   field=cfg.datamodule.field,
                                   mode=cfg.model.name,
                                   every_n_epoch=cfg.val_every_n_epoch)
    callbacks.append(plots_callback)

    trainer: Trainer = instantiate(cfg.trainer, logger=wandb_logger, callbacks=callbacks)

    trainer.fit(task,
                datamodule=datamodule)

    wandb.finish()


if __name__ == "__main__":
    train()

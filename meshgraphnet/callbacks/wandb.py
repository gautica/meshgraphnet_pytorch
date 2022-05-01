from pathlib import Path
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint


class WandbModelCheckpoint(ModelCheckpoint):
    """Save checkpoints into the W&B run directory to sync them automatically."""

    def __init__(self, **kwargs):
        run_dir = Path(wandb.run.dir)
        cp_dir = run_dir / "checkpoints"

        super().__init__(**kwargs, dirpath=str(cp_dir))
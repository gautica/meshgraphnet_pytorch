import pytorch_lightning as pl
from meshgraphnet.data.dataset import MeshDataset
from torch_geometric.loader import DataLoader


class MeshDataModule(pl.LightningDataModule):
    def __init__(self, config, transform=None):
        super(MeshDataModule, self).__init__()
        self.config = config

    def prepare_data(self):
        print("prepare data ...")

    def setup(self, stage=None):
        """ how to split data """
        print("setup dataset ...")
        if stage == "fit" or stage is None:
            self.train_ds = MeshDataset(self.config, split="train")
            self.valid_ds = MeshDataset(self.config, split="valid")
            print("Number of data: ", len(self.train_ds))
            print("Number of data: ", len(self.valid_ds))

    def transfer_batch_to_device(self, batch, device, dataloader_idx: int):
        batch = batch.to(device)
        return batch

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.config.batch_size_train, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.config.batch_size_valid, shuffle=False, num_workers=8)

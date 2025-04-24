from pathlib import Path
from typing import Any, Dict

import hydra
import numpy as np
import polars as pl
import torch
from lightning.pytorch import LightningDataModule
from src.conf import InferenceConfig, TrainConfig
from src.utils.common import load_from_yaml
from torch.utils.data import DataLoader, Dataset


class CIBMTRDataset(Dataset):
    def __init__(
        self,
        df: pl.DataFrame,
        cat_col: list[str],
        num_col: list[str],
        target_col: str | None = None,
    ) -> None:
        super().__init__()
        self.cat_col = cat_col
        self.num_col = num_col
        self.target_col = target_col


        self.cat_df = df.select(self.cat_col)
        self.num_df = df.select(self.num_col)
        self.id = df.select("ID")
        
        if self.target_col is not None:
            self.target_df = df.select(self.target_col)
            self.efs = df.select("efs")
            self.efs_time = df.select("efs_time")
            self.race_group = df.select("race_group")

    def __len__(self):
        return len(self.cat_df)

    def __getitem__(self, idx) -> Dict[str, Any]:
        cat_df = self.cat_df.row(idx)
        num_df = self.num_df.row(idx)
        id = self.id.row(idx)
        

        if self.target_col is not None:
            efs = self.efs.row(idx)
            efs_time = self.efs_time.row(idx)
            race_group = self.race_group.row(idx)
            target_df = self.target_df.row(idx)
            data = {
                "ID": id,
                "cat_col": torch.tensor(np.array(cat_df), dtype=torch.int32),
                "num_col": torch.tensor(np.array(num_df).astype(np.float32), dtype=torch.float32),
                "target": torch.tensor(np.array(target_df), dtype=torch.float32),
                "efs": efs,
                "efs_time": efs_time,
                "race_group": race_group,
            }
        else:
            data = {
                "ID": id,
                "cat_col": torch.tensor(np.array(cat_df), dtype=torch.int32),
                "num_col": torch.tensor(np.array(num_df).astype(np.float32), dtype=torch.float32),
                # "target": None,
                # "efs": None,
                # "efs_time": None,
                # "race_group": None,
            }
        return data

class CIBMTRDataModule(LightningDataModule):
    def __init__(
        self,
        cfg: TrainConfig | InferenceConfig,
        fold: int | None = None,
        n_repeat: int | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.fold = fold
        self.n_repeat = n_repeat
        self.data = load_from_yaml(Path(self.cfg.dir.processed_dir) / f"{cfg.exp_name}_processed_data.yaml")

        if (self.fold is not None) & (self.n_repeat is not None) & (self.cfg.phase == "train"): # train
            self.train_df = (
                pl.scan_csv(Path(cfg.dir.processed_dir) / "train.csv")
                .filter(pl.col(f"fold_repeat{self.n_repeat}") != self.fold)
            ).collect()
            self.val_df = (
                pl.scan_csv(Path(cfg.dir.processed_dir) / "train.csv")
                .filter(pl.col(f"fold_repeat{self.n_repeat}") == self.fold)
            ).collect()
        elif (self.fold is None) & (self.n_repeat is None) & (self.cfg.phase == "test"): # test
            self.test_df = (
                pl.scan_csv(Path(cfg.dir.processed_dir) / "test.csv")
            ).collect()
        else:
            raise ValueError(f"incorrect fold({self.fold}) or cfg.phase({self.cfg.phase})")


        if self.cfg.phase == "train":
            self.train_dataset = CIBMTRDataset(
                df=self.train_df,
                cat_col=self.data["CATS"],
                num_col=self.data["NUMS"],
                target_col=self.cfg.target_col,
            )

            self.val_dataset = CIBMTRDataset(
                df=self.val_df,
                cat_col=self.data["CATS"],
                num_col=self.data["NUMS"],
                target_col=self.cfg.target_col,
            )
        elif self.cfg.phase == "test":
            self.test_dataset = CIBMTRDataset(
                df=self.test_df,
                cat_col=self.data["CATS"],
                num_col=self.data["NUMS"],
            )
        else:
            raise ValueError(f"incorrect cfg.phase :{self.cfg.phase}")

    def train_dataloader(self) -> DataLoader:
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.cfg.dataset.batch_size,
            shuffle=True,
            num_workers=self.cfg.dataset.num_workers,
            pin_memory=True,
        )
        return train_loader

    def val_dataloader(self) -> DataLoader:
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.cfg.dataset.batch_size,
            shuffle=False,
            num_workers=self.cfg.dataset.num_workers,
            pin_memory=True,
        )
        return val_loader

    def test_dataloader(self) -> DataLoader:
        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.cfg.dataset.batch_size,
            shuffle=False,
            num_workers=self.cfg.dataset.num_workers,
            pin_memory=True,
        )
        return test_loader
        


@hydra.main(config_path="../run/conf", config_name="train", version_base="1.3")
def main(cfg: TrainConfig) -> None:

    datamodule = CIBMTRDataModule(cfg=cfg, fold=0, n_repeat=0)
    print(datamodule)

    dl = datamodule.train_dataloader()
    for i, data in enumerate(dl):
        print(f"{i}: {data}")

if __name__ == "__main__":
    main()

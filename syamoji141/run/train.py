import gc
import logging
import os
from pathlib import Path
from typing import List

import hydra
import polars as pl
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint, RichModelSummary,
                                         RichProgressBar)
from lightning.pytorch.loggers import WandbLogger
from src.conf import TrainConfig
from src.datamodule import CIBMTRDataModule
from src.modelmodule import CIBMTRModel
from src.utils.common import score

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s"
)
LOGGER = logging.getLogger(Path(__file__).name)

@hydra.main(config_path="conf", config_name="train", version_base="1.3")
def main(cfg: TrainConfig):
    seed_everything(cfg.seed)

    model_save_dir: Path = Path(cfg.dir.model_dir) / cfg.exp_name
    os.makedirs(model_save_dir, exist_ok=True)

    pl_logger = WandbLogger(
        name=cfg.exp_name, project="CIBMTR",
        entity="gaiji",
        offline=cfg.offline,
        save_dir=cfg.dir.output_dir,
        checkpoint_name=cfg.exp_name,
    )
    pl_logger.log_hyperparams(cfg)

    oof_df_list: List[pl.DataFrame] = []

    for repeat in range(cfg.n_repeats):
        for fold in range(cfg.folds):
            LOGGER.info(f"Start Training Fold {fold} in repeat {repeat}")
            datamodule = CIBMTRDataModule(cfg, fold, repeat)
            LOGGER.info("Set Up DataModule")
            model = CIBMTRModel(cfg, fold, repeat)

            checkpoint_cb = ModelCheckpoint(
                verbose=True,
                monitor=f"fold{fold}_val_loss",
                mode=cfg.trainer.monitor_mode,
                save_top_k=1,
                save_last=False,
            )
            lr_monitor = LearningRateMonitor("epoch")
            progress_bar = RichProgressBar()
            model_summary = RichModelSummary(max_depth=2)
            early_stopping = EarlyStopping(
                monitor=f"fold{fold}_val_loss",
                mode=cfg.trainer.monitor_mode,
                patience=cfg.trainer.patience,
            )

            trainer = Trainer(
                default_root_dir=cfg.dir.output_dir,
                accelerator=cfg.trainer.accelerator,
                precision=16 if cfg.trainer.use_amp else 32,
                # training
                max_epochs=1 if cfg.debug else cfg.trainer.epochs,
                max_steps=cfg.trainer.epochs * len(datamodule.train_dataloader()),
                gradient_clip_val=cfg.trainer.gradient_clip_val,
                accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
                callbacks=[checkpoint_cb, lr_monitor, progress_bar, model_summary, early_stopping],
                logger=pl_logger,
                num_sanity_val_steps=0,
                log_every_n_steps=int(len(datamodule.train_dataloader()) * 0.1),
                reload_dataloaders_every_n_epochs=1,
                sync_batchnorm=True,
                check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
            )

            trainer.fit(model=model, datamodule=datamodule)

            LOGGER.info("Infer OOF")
            val_loader = datamodule.val_dataloader()
            predictions = trainer.predict(model, val_loader)
            # predictionsはList[Tensor]になるのでまとめる
            if cfg.model.name == "tabm":
                pred_tensor = torch.cat(predictions, dim=0).squeeze(-1).mean(1).to(torch.float32).cpu().numpy().flatten()
            else:
                pred_tensor = torch.cat(predictions, dim=0).squeeze(-1).to(torch.float32).cpu().numpy().flatten()

            val_ids = []
            val_efs = []
            val_efs_time = []
            val_race_group = []

            for batch in val_loader:
                val_ids.extend(batch["ID"])
                val_efs.extend(batch["efs"])
                val_efs_time.extend(batch["efs_time"])
                val_race_group.extend(batch["race_group"])

            val_ids = torch.cat(val_ids).squeeze(-1).cpu().numpy()
            val_efs = torch.cat(val_efs).to(torch.int32).squeeze(-1).cpu().numpy()
            val_efs_time = torch.cat(val_efs_time).squeeze(-1).cpu().numpy()
            val_race_group = torch.cat(val_race_group).squeeze(-1).cpu().numpy()

            fold_oof_df = pl.DataFrame({
                "ID": val_ids,
                "efs": val_efs,
                "efs_time": val_efs_time,
                "race_group": val_race_group,
                "prediction": pred_tensor
            })
            oof_df_list.append(fold_oof_df)

            del model
            del trainer
            del datamodule
            gc.collect()

    oof_df = pl.concat(oof_df_list, how="vertical")

    oof_df_ens = (
        oof_df.group_by("ID", maintain_order=True)
        .agg(
            [
                pl.col("efs").first().alias("efs"),
                pl.col("efs_time").first().alias("efs_time"),
                pl.col("race_group").first().alias("race_group"),
                pl.col("prediction").mean().alias("prediction"),
            ]
        )
        .sort("ID") 
    )

    oof_df_solution = oof_df_ens.select(
        ["ID", "efs", "efs_time", "race_group"]
    ).to_pandas()
    oof_df_prediction = oof_df_ens.select(["ID", "prediction"]).to_pandas()

    oof_score, metric_list, race_keys = score(
        oof_df_solution.copy(),
        oof_df_prediction.copy(),
        row_id_column_name="ID",
    )

    LOGGER.info(f"OOF CV_c-index: {oof_score}")
    pl_logger.log_metrics({"CV_c-index": oof_score})

if __name__ == "__main__":
    main()
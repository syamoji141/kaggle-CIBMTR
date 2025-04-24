import logging
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from lightning.pytorch import LightningModule
from src.conf import InferenceConfig, TrainConfig
from src.model.common import get_model
from src.utils.common import score
from torchsurv.loss.cox import neg_partial_log_likelihood
from transformers import get_cosine_schedule_with_warmup

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s"
)
LOGGER = logging.getLogger(Path(__file__).name)

class CIBMTRModel(LightningModule):
    def __init__(self, 
    cfg: TrainConfig | InferenceConfig,
    fold: int,
    n_repeat: int,
    ):
        super().__init__()
        self.cfg = cfg
        self.model = get_model(self.cfg, fold, n_repeat)
        self.validation_step_outputs = []
        self.__best_loss = np.inf
        self.fold = fold
        self.n_repeat = n_repeat


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch):
        cat = batch["cat_col"]
        num = batch["num_col"]
        target = batch["target"]
        if self.cfg.model.name == "tabm":
            if self.cfg.repeat_interleave:
                target = target.repeat_interleave(self.cfg.model.params.k)
                pred = self.model(cat, num).flatten(0, 1)
            else:
                pred = self.model(cat, num).mean(1)
        else:
            pred = self.model(cat, num)

        if self.cfg.model.params.multi_output:
            pred, pred_efs_time = self.model(cat, num)
            efs_time = batch["efs_time"][0].to(torch.float32)
            if self.cfg.loss == "mse":
                loss1 = F.mse_loss(pred, target, reduction="none")
                loss1 = loss1.mean()
                loss2 = F.mse_loss(pred_efs_time, efs_time, reduction="none")
                loss2 = loss2.mean()
                loss = self.cfg.alpha_1 * loss1 + self.cfg.alpha_2 * loss2
            elif self.cfg.loss == "rmse":
                loss1 = torch.sqrt(F.mse_loss(pred, target, reduction="none"))
                loss1 = loss1.mean()
                loss2 = torch.sqrt(F.mse_loss(pred_efs_time, efs_time, reduction="none"))
                loss2 = loss2.mean()
                loss = self.cfg.alpha_1 * loss1 + self.cfg.alpha_2 * loss2
            elif self.cfg.loss == "cox":
                efs = batch["efs"][0].to(torch.int32)
                loss1 = neg_partial_log_likelihood(pred, event=efs, time=efs_time, reduction="mean")
                loss2 = F.mse_loss(pred_efs_time, efs_time, reduction="none")
                loss2 = loss2.mean()
                loss = self.cfg.alpha_1 * loss1 + self.cfg.alpha_2 * loss2
        elif self.cfg.loss == "mse":
            loss = F.mse_loss(pred, target, reduction="none")
            loss = loss.mean()
        elif self.cfg.loss == "rmse":
            loss = torch.sqrt(F.mse_loss(pred, target, reduction="none"))
            loss = loss.mean()
        elif self.cfg.loss == "cox":
            efs = batch["efs"][0].to(torch.int32)
            efs_time = batch["efs_time"][0]
            loss = neg_partial_log_likelihood(pred, event=efs, time=efs_time, reduction="mean")
        else:
            raise ValueError(f"Invalid loss name: {self.cfg.loss}")

        # if self.n_repeat == 0:
        self.log(f"fold{self.fold}_train_loss", loss, on_step=False, on_epoch=True, logger=True, batch_size=cat.size(0), prog_bar=True)
        # else:
        #     LOGGER.info(f"fold{self.fold}_train_loss: {loss}")
        return loss

    def validation_step(self, batch):
        id = batch["ID"]
        cat = batch["cat_col"]
        num = batch["num_col"]
        target = batch["target"]
        if self.cfg.model.name == "tabm":
            if self.cfg.repeat_interleave:
                pred = self.model(cat, num).flatten(0, 1)
            else:
                pred = self.model(cat, num).mean(1)
        else:
            pred = self.model(cat, num)
        efs = batch["efs"]
        efs_time = batch["efs_time"]
        race_group = batch["race_group"]

        if self.cfg.model.params.multi_output:
            pred, pred_efs_time = self.model(cat, num)
            _efs_time = batch["efs_time"][0].to(torch.float32)
            if self.cfg.loss == "mse":
                loss1 = F.mse_loss(pred, target, reduction="none")
                loss1 = loss1.mean()
                loss2 = F.mse_loss(pred_efs_time, _efs_time, reduction="none")
                loss2 = loss2.mean()
                loss = self.cfg.alpha_1 * loss1 + self.cfg.alpha_2 * loss2
            elif self.cfg.loss == "rmse":
                loss1 = torch.sqrt(F.mse_loss(pred, target, reduction="none"))
                loss1 = loss1.mean()
                loss2 = torch.sqrt(F.mse_loss(pred_efs_time, _efs_time, reduction="none"))
                loss2 = loss2.mean()
                loss = self.cfg.alpha_1 * loss1 + self.cfg.alpha_2 * loss2
            elif self.cfg.loss == "cox":
                _efs = batch["efs"][0].to(torch.int32)
                loss1 = neg_partial_log_likelihood(pred, event=_efs, time=_efs_time, reduction="mean")
                loss2 = F.mse_loss(pred_efs_time, _efs_time, reduction="none")
                loss2 = loss2.mean()
                loss = self.cfg.alpha_1 * loss1 + self.cfg.alpha_2 * loss2
        elif self.cfg.loss == "mse":
            pred = self.model(cat, num)
            if (self.cfg.repeat_interleave) & (self.cfg.model.name == "tabm"):
                target = target.repeat_interleave(self.cfg.model.params.k)
                loss = F.mse_loss(pred.flatten(0, 1), target, reduction="none")
            elif ~(self.cfg.repeat_interleave) & (self.cfg.model.name == "tabm"):
                loss = F.mse_loss(pred.mean(1), target, reduction="none")
            else:
                loss = torch.sqrt(F.mse_loss(pred, target, reduction="none"))
            loss = F.mse_loss(pred, target, reduction="none")
            loss = loss.mean()
        elif self.cfg.loss == "rmse":
            pred = self.model(cat, num)
            if (self.cfg.repeat_interleave) & (self.cfg.model.name == "tabm"):
                target = target.repeat_interleave(self.cfg.model.params.k)
                loss = torch.sqrt(F.mse_loss(pred.flatten(0, 1), target, reduction="none"))
            elif ~(self.cfg.repeat_interleave) & (self.cfg.model.name == "tabm"):
                loss = torch.sqrt(F.mse_loss(pred.mean(1), target, reduction="none"))
            else:
                loss = torch.sqrt(F.mse_loss(pred, target, reduction="none"))
            loss = loss.mean()
        elif self.cfg.loss == "cox":
            pred = self.model(cat, num)
            _efs = efs[0].to(torch.int32)
            _efs_time = efs_time[0]
            loss = neg_partial_log_likelihood(pred, event=_efs, time=_efs_time, reduction="mean")
        else:
            raise ValueError(f"Invalid loss name: {self.cfg.loss}")
        
        # if self.n_repeat == 0:
        self.log(f"fold{self.fold}_val_loss", loss, on_step=False, on_epoch=True, logger=True, batch_size=cat.size(0), prog_bar=True)
        # else:
        #     LOGGER.info(f"fold{self.fold}_val_loss: {loss}")
        self.validation_step_outputs.append((id, pred, target, loss, efs, efs_time, race_group))
        return loss

    def on_validation_epoch_end(self) -> None:
        ids = []
        efses = torch.cat([x[4][0] for x in self.validation_step_outputs]).cpu().numpy()
        efs_times = torch.cat([x[5][0] for x in self.validation_step_outputs]).cpu().numpy()
        race_groups = torch.cat([x[6][0] for x in self.validation_step_outputs]).cpu().numpy()
        # for x in self.validation_step_outputs:
        #     ids.extend(x[0])

        ids = torch.cat([x[0][0] for x in self.validation_step_outputs]).cpu().numpy()

        # ids = torch.cat([x[0] for x in self.validation_step_outputs]).cpu().numpy()
        if self.trainer.sanity_checking:
            pred = torch.cat([x[1] for x in self.validation_step_outputs]).squeeze(-1).cpu().numpy()
        else:
            pred = torch.cat([x[1] for x in self.validation_step_outputs]).squeeze(-1).to(torch.float32).cpu().numpy()

            losses = torch.cat([torch.tensor([x[3]]) for x in self.validation_step_outputs]).to(torch.float32).cpu().numpy()
            loss = losses.mean()

            solution = pl.DataFrame(
                {
                    "ID": ids,
                    "efs": efses,
                    "efs_time": efs_times,
                    "race_group": race_groups,
                }
            ).to_pandas()

            pred_df = pl.DataFrame(
                {
                    "ID": ids,
                    "prediction": pred,
                }
            ).to_pandas()

            c_index, metric_list, races   = score(solution=solution, submission=pred_df, row_id_column_name="ID")
            for i, race in enumerate(races):
                self.log(f"fold{self.fold}_race{race}_concordance", metric_list[i], on_step=False, on_epoch=True, logger=True, prog_bar=True)
            self.log(f"fold{self.fold}_race_mean_concordance", np.mean(metric_list), on_step=False, on_epoch=True, logger=True, prog_bar=True)
            self.log(f"fold{self.fold}_race_std_concordance", np.sqrt(np.var(metric_list)), on_step=False, on_epoch=True, logger=True, prog_bar=True)
            # if self.n_repeat == 0:
            self.log(f"fold{self.fold}_c-index", c_index, on_step=False, on_epoch=True, logger=True, prog_bar=True)
            # else:
            #     LOGGER.info(f"fold{self.fold}_c-index: {c_index}")

        if loss < self.__best_loss:
            torch.save(self.model.state_dict(), Path(self.cfg.dir.model_dir) / f"{self.cfg.exp_name}_best_model_fold{self.fold}_n_repeat{self.n_repeat}.pth")
            print(f"Saved best model {self.__best_loss} -> {loss}")
            self.__best_loss = loss
        self.validation_step_outputs.clear()

    def predict_step(self, batch):
        cat = batch["cat_col"]
        num = batch["num_col"]
        if self.cfg.model.params.multi_output:
            pred, _ = self.model(cat, num)
        else:
            pred= self.model(cat, num)
        return pred

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.cfg.optimizer.lr, weight_decay=self.cfg.optimizer.weight_decay)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_training_steps=self.trainer.max_steps, **self.cfg.scheduler
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
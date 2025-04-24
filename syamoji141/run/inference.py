import logging
from pathlib import Path

import hydra
import numpy as np
import polars as pl
import torch
from src.conf import InferenceConfig
from src.datamodule import CIBMTRDataModule
from src.model.common import get_model
from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s"
)
LOGGER = logging.getLogger(Path(__file__).name)

@hydra.main(config_path="conf", config_name="inference", version_base="1.3")
def main(cfg: InferenceConfig):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    total_preds = None
    for repeat in range(cfg.n_repeats):
        for fold in range(cfg.folds):
            LOGGER.info(f"Start Inference Fold {fold} in repeat {repeat}")
            datamodule = CIBMTRDataModule(cfg, None, None)

            LOGGER.info("Set Up DataModule")

            model = get_model(cfg)
            model.to(device)

            model.load_state_dict(
            torch.load(Path(cfg.dir.model_dir) / f"{cfg.exp_name}_best_model_fold{fold}_n_repeat{repeat}.pth", weights_only=True),
            # strict=False,  #  Unexpected key(s) in state_dictの回避
            )
            model.eval()

            preds = []
            ids = []
            for batch in tqdm(datamodule.test_dataloader()):
                with torch.no_grad():
                    with torch.autocast(device_type=device_str, enabled=cfg.use_amp):
                        cat_col = batch["cat_col"].to(device)
                        num_col = batch["num_col"].to(device)
                        id = batch["ID"]
                        pred = model(cat_col, num_col)
                    
                    ids.append(np.array(id))
                    preds.append(pred.detach().to(torch.float32).squeeze(-1).cpu().numpy())

            ids = np.concatenate(ids).flatten()
            preds = np.concatenate(preds)

            if total_preds is None:
                total_ids = ids
                total_preds = preds
            else:
                total_preds += preds

    total_preds /= (cfg.folds * cfg.n_repeats)

    sub_df = pl.DataFrame(
        {
            "ID": total_ids,
            "prediction": total_preds
        }
    )

    sub_df.write_csv(Path(cfg.dir.sub_dir) / "submission.csv")

    print(sub_df)




if __name__ == "__main__":
    main()
            
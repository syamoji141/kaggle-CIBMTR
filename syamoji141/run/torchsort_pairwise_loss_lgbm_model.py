import hydra
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import wandb
from wandb.integration.lightgbm import wandb_callback
from src.conf import PrepareDataConfig
import warnings
from lifelines.utils import concordance_index
import time
import torch  
import torchsort
from typing import Tuple
import pickle  

warnings.filterwarnings("ignore")
pd.options.display.max_columns = None

all_model_scores = {}

def get_custom_logistic_loss(race_group_arr: np.ndarray, event_arr: np.ndarray, sc_loss_weight: float = 1.0):
    def custom_logistic_loss(pred: np.ndarray, dataset: lgb.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        label = dataset.get_label() 
        n = len(label)
        if pred.shape[0] != n:
            pred = pred.reshape(n, -1)[:, 0]
        if np.all(pred == pred[0]):
            print("Fallback: Calculating RMSE Loss")
            grad = pred - label
            hess = np.ones_like(label)
            return grad, hess

        pred_tensor = torch.tensor(pred, dtype=torch.float64, requires_grad=True)
        target_tensor = torch.tensor(label, dtype=torch.float64) 
        event_tensor = torch.tensor(event_arr, dtype=torch.int64)
        if torch.cuda.is_available():
            pred_tensor = pred_tensor.to("cuda")
            target_tensor = target_tensor.to("cuda")
            event_tensor = event_tensor.to("cuda")

        unique_races = np.unique(race_group_arr)
        group_loss_list = []
        for rg in unique_races:
            mask = (race_group_arr == rg)
            group_mask = torch.tensor(mask, dtype=torch.bool, device=pred_tensor.device)
            pred_group = pred_tensor[group_mask]
            target_group = target_tensor[group_mask]
            event_group = event_tensor[group_mask] 
            if pred_group.shape[0] < 2:
                continue

            pred_rank = torchsort.soft_rank(pred_group.unsqueeze(0)).squeeze(0)
            target_rank = torchsort.soft_rank(target_group.unsqueeze(0)).squeeze(0)

            diff_pred = pred_rank.unsqueeze(0) - pred_rank.unsqueeze(1)
            diff_target = target_rank.unsqueeze(0) - target_rank.unsqueeze(1)
            base_mask = (diff_target != 0)

            event_left = event_group.unsqueeze(0)  # (n, 1)
            event_right = event_group.unsqueeze(1) # (1, n)
            time_left = target_group.unsqueeze(0)    # (n, 1)
            time_right = target_group.unsqueeze(1)   # (1, n)

            cond1 = (event_left == 1) & (event_right == 1)
            cond2 = ((event_left == 0) & (event_right == 1) & (time_left > time_right))
            cond3 = ((event_left == 1) & (event_right == 0) & (time_left < time_right))
            valid_event_mask = cond1 | cond2 | cond3
            valid_mask = base_mask & valid_event_mask
            if valid_mask.sum() == 0:
                continue

            pair_labels = - torch.sign(time_left - time_right)
            pair_loss = torch.log1p(torch.exp(- pair_labels * diff_pred))
            group_loss = pair_loss[valid_mask].mean()
            group_loss_list.append(group_loss)
        
        if len(group_loss_list) == 0:
            print("Fallback: Calculating RMSE Loss")
            grad = pred - label
            hess = np.ones_like(label)
            return grad, hess

        group_losses = torch.stack(group_loss_list)
        loss = (group_losses.mean() + sc_loss_weight * group_losses.std()) * 10000

        grad_tensor = torch.autograd.grad(loss, pred_tensor, create_graph=True)[0]
        grad = grad_tensor.cpu().detach().numpy()
        hess = np.ones_like(pred)
        return grad, hess
    return custom_logistic_loss

def get_custom_logistic_feval(race_group_arr: np.ndarray, event_arr: np.ndarray, sc_loss_weight: float = 1.0):
    def custom_logistic_feval(pred: np.ndarray, dataset: lgb.Dataset):
        label = dataset.get_label()  # plain な efs_time
        n = len(label)
        if pred.shape[0] != n:
            pred = pred.reshape(n, -1)[:, 0]

        pred_tensor = torch.tensor(pred, dtype=torch.float64)
        target_tensor = torch.tensor(label, dtype=torch.float64)
        event_tensor = torch.tensor(event_arr, dtype=torch.int64)
                                
        unique_races = np.unique(race_group_arr)
        group_loss_list = []
        for rg in unique_races:
            mask = (race_group_arr == rg)
            group_mask = torch.tensor(mask, dtype=torch.bool)
            pred_group = pred_tensor[group_mask]
            target_group = target_tensor[group_mask]
            event_group = event_tensor[group_mask]
            if pred_group.shape[0] < 2:
                continue

            pred_rank = torchsort.soft_rank(pred_group.unsqueeze(0)).squeeze(0)
            target_rank = torchsort.soft_rank(target_group.unsqueeze(0)).squeeze(0)
            diff_pred = pred_rank.unsqueeze(0) - pred_rank.unsqueeze(1)
            diff_target = target_rank.unsqueeze(0) - target_rank.unsqueeze(1)
            base_mask = (diff_target != 0)
            event_left = event_group.unsqueeze(0)
            event_right = event_group.unsqueeze(1)
            time_left = target_group.unsqueeze(0)
            time_right = target_group.unsqueeze(1)
            cond1 = (event_left == 1) & (event_right == 1)
            cond2 = ((event_left == 0) & (event_right == 1) & (time_left > time_right))
            cond3 = ((event_left == 1) & (event_right == 0) & (time_left < time_right))
            valid_event_mask = cond1 | cond2 | cond3
            valid_mask = base_mask & valid_event_mask
            if valid_mask.sum() == 0:
                continue
            pair_labels = - torch.sign(time_left - time_right)
            pair_loss = torch.log1p(torch.exp(- pair_labels * diff_pred))
            group_loss = pair_loss[valid_mask].mean()
            group_loss_list.append(group_loss.item())
        if len(group_loss_list) == 0:
            overall_loss = 0.0
        else:
            overall_loss = (np.mean(group_loss_list) + sc_loss_weight * np.std(group_loss_list)) * 10000
        return "sc_logistic_loss", overall_loss, False
    return custom_logistic_feval

@hydra.main(config_path="conf", config_name="pairwise_loss", version_base="1.3")
def main(cfg: PrepareDataConfig):
    wandb.init(
        project="CIBMTR",
        name=f"{cfg.exp_name}_lgbm",
        entity="gaiji",
        notes=cfg.note,
    )
    wandb.run.config._allow_val_change = True

    train = pd.read_csv(Path(cfg.dir.processed_dir) / "train.csv", index_col="ID")
    test = pd.read_csv(Path(cfg.dir.processed_dir) / "test.csv", index_col="ID")

    features = [f for f in test.columns if f != "ID"]
    cat_features = list(train.select_dtypes(object).columns)
    train[cat_features] = train[cat_features].astype(str).astype("category")
    race_groups = np.unique(train.race_group)

    kf = StratifiedKFold(shuffle=True, random_state=42)
    all_scores = []

    def evaluate_fold(y_va_pred, fold, X_va, idx_va):
        metric_list = []
        for race in race_groups:
            mask = X_va.race_group.values == race
            c_index_race = concordance_index(
                train.efs_time.iloc[idx_va][mask],
                -y_va_pred[mask],
                train.efs.iloc[idx_va][mask],
            )
            metric_list.append(c_index_race)
        fold_score = np.mean(metric_list) - np.std(metric_list)
        print(f"# Total fold {fold}: {fold_score:.3f} (mean={np.mean(metric_list):.3f}, std={np.std(metric_list):.3f})")
        wandb.log({f"fold_{fold}_score": fold_score})
        all_scores.append(metric_list)

    def display_overall(label):
        df = pd.DataFrame(all_scores, columns=race_groups)
        df["mean"] = df[race_groups].mean(axis=1)
        df["std"] = np.std(df[race_groups], axis=1)
        df["score"] = df["mean"] - df["std"]
        df = df.T
        df["Overall"] = df.mean(axis=1)
        temp = df.drop(index=["std"]).values
        overall = df.loc["score", "Overall"]
        print(f"# Overall: {overall:.3f} {label}")
        all_model_scores[label] = overall
        display(
            df.iloc[: len(race_groups)]
            .style.format(precision=3)
            .background_gradient(axis=None, vmin=temp.min(), vmax=temp.max(), cmap="cool")
            .concat(df.iloc[len(race_groups):].style.format(precision=3))
        )
        wandb.log({"overall_score": overall})

    true_value = train.efs_time.values
    pred_value = np.zeros(len(train))

    start = time.time()
    for fold, (idx_tr, idx_va) in enumerate(kf.split(train, train.race_group.astype(str))):
        X_tr = train.iloc[idx_tr][features]
        X_va = train.iloc[idx_va][features]
        y_tr = train.iloc[idx_tr].efs_time.values
        y_va = train.iloc[idx_va].efs_time.values

        custom_loss = get_custom_logistic_loss(train.iloc[idx_tr].race_group.values,
                                                train.iloc[idx_tr].efs.values,
                                                sc_loss_weight=cfg.sc_loss_weight)
        custom_feval = get_custom_logistic_feval(train.iloc[idx_va].race_group.values,
                                                  train.iloc[idx_va].efs.values,
                                                  sc_loss_weight=cfg.sc_loss_weight)

        dtrain_reg = lgb.Dataset(
            X_tr,
            label=y_tr,
            categorical_feature=cat_features,
            free_raw_data=False,
        )
        dvalid_reg = lgb.Dataset(
            X_va,
            label=y_va,
            reference=dtrain_reg,
            categorical_feature=cat_features,
        )
        regressor_params = {
            "objective": custom_loss,
            "boosting": "gbdt",
            "num_leaves": 31,
            "max_depth": 4,
            "learning_rate": 0.02,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.5,
            "min_child_weight": 80,
            "force_col_wise": True,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "max_bin": 255,
            "seed": 42,
            "verbose": -1,
        }
        reg_model = lgb.train(
            regressor_params,
            dtrain_reg,
            num_boost_round=100000,
            valid_sets=[dvalid_reg],
            valid_names=["valid"],
            callbacks=[lgb.log_evaluation(100),
                       lgb.early_stopping(100),
                       wandb_callback()],
            feval=custom_feval,
        )
        y_va_pred = reg_model.predict(X_va)
        pred_value[idx_va] = y_va_pred

        evaluate_fold(y_va_pred, fold, X_va, idx_va)

        if cfg.model_dump:
            model_dump_dir = Path(cfg.model_dump_dir)
            model_dump_dir.mkdir(parents=True, exist_ok=True)
            model_filename = model_dump_dir / f"model_{cfg.exp_name}_fold{fold}.pkl"
            with open(model_filename, "wb") as f:
                pickle.dump(reg_model, f)
            print(f"Saved model for fold {fold} as {model_filename}")

    end = time.time()
    time_diff = end - start 
    print("Total CV time: {:.2f} seconds".format(time_diff))

    display_overall(label="Plain efs_time LightGBM (SC-index Optimized)")

    true_alive = true_value[train.efs.values == 0]
    true_dead = true_value[train.efs.values == 1]
    pred_alive = pred_value[train.efs.values == 0]
    pred_dead = pred_value[train.efs.values == 1]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(true_alive, bins=30, color="b", alpha=0.5, label="TrueAlive")
    axes[0].hist(true_dead, bins=30, color="r", alpha=0.5, label="TrueDead")
    axes[0].set_title("TrueValue (efs_time)")
    axes[0].legend()

    axes[1].hist(pred_alive, bins=30, color="b", alpha=0.5, label="PredAlive")
    axes[1].hist(pred_dead, bins=30, color="g", alpha=0.5, label="PredDead")
    axes[1].hist(pred_value, bins=30, color="y", alpha=0.5, label="AllPredValue")
    axes[1].set_title("Predicted Value")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    wandb.finish()

if __name__ == "__main__":
    main()

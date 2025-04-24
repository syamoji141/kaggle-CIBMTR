from pathlib import Path

import polars as pl
import rtdl_num_embeddings
import torch
from src.conf import InferenceConfig, TrainConfig
from src.model.mlp import MLP
from src.model.tabm_reference import Model
from src.model.transformer import TableTransformer
from src.utils.common import load_from_yaml


def get_model(cfg: TrainConfig, fold: int, n_repeat: int):
    if cfg.model.name == "mlp":
        data = load_from_yaml(Path(cfg.dir.processed_dir) / f"{cfg.exp_name}_processed_data.yaml")

        model = MLP(
            cat_size=data["CAT_SIZE"],
            cat_emb=data["CAT_EMB"],
            num_features=len(data["NUMS"]),
        )
    elif cfg.model.name == "transformer":
        data = load_from_yaml(Path(cfg.dir.processed_dir) / f"{cfg.exp_name}_processed_data.yaml")

        model = TableTransformer(
            cat_sizes=data["CAT_SIZE"],
            cat_emb_dims=data["CAT_EMB"],
            num_features=len(data["NUMS"]),
            d_model=cfg.model.params.d_model,
            nhead=cfg.model.params.nhead,
            num_layers=cfg.model.params.num_layers,
            dropout=cfg.model.params.dropout,
            use_batch_first=True,
            multi_output=cfg.model.params.multi_output,
        )
    elif cfg.model.name == "tabm":
        data = load_from_yaml(Path(cfg.dir.processed_dir) / f"{cfg.exp_name}_processed_data.yaml")
        if (fold is None) and (n_repeat is None):
            torch.load(bins, Path(cfg.dir.model_dir) / f"{cfg.exp_name}_fold{fold}_n_repeat{fold}.pt")
        else:
            bins = (
                pl.scan_csv(Path(cfg.dir.processed_dir) / "train.csv")
                .filter(pl.col(f"fold_repeat{n_repeat}") != fold)
                .select(data["NUMS"])
            ).collect().to_numpy()
            bins = rtdl_num_embeddings.compute_bins(torch.tensor(bins, dtype=torch.float32))
        if (fold is not None) and (n_repeat is not None):
            torch.save(bins, Path(cfg.dir.model_dir) / f"{cfg.exp_name}_fold{fold}_n_repeat{fold}.pt")

        model = Model(
            n_num_features=len(data["NUMS"]),
            cat_cardinalities=data["CAT_SIZE"],
            n_classes=None,
            backbone={
            'type': cfg.model.params.type,
            'n_blocks': cfg.model.params.n_blocks,
            'd_block': cfg.model.params.d_block,
            'dropout': cfg.model.params.dropout,
            },
            bins=bins,
            num_embeddings=(
                None
                if bins is None
                else {
                    'type': 'PiecewiseLinearEmbeddings',
                    'd_embedding': cfg.model.params.d_embedding,
                    'activation': True,
                    'version': 'B',
                }
            ),
            arch_type=cfg.model.params.arch_type,
            k=cfg.model.params.k,
        )
    else:
        raise ValueError(f"Invalid model name: {cfg.model.name}")
    
    return model
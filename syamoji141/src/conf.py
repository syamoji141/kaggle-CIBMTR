from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class DirConfig:
    data_dir: str
    processed_dir: str
    output_dir: str
    model_dir: str
    sub_dir: str


@dataclass
class PrepareDataConfig:
    dir: DirConfig
    exp_name: str
    phase: str
    encoder: str
    repeats: int
    folds: int
    batch_size: int
    target_type: str

@dataclass
class DatasetConfig:
    batch_size: int
    num_workers: int

@dataclass
class ModelConfig:
    name: str
    params: Dict[str, Any]

@dataclass
class SchedulerConfig:
    num_warmup_steps: int

@dataclass
class OptimizerConfig:
    lr: float
    weight_decay: float

@dataclass
class TrainerConfig:
    epochs: int
    accelerator: str
    use_amp: bool
    debug: bool
    seed: int
    gradient_clip_val: float
    accumulate_grad_batches: int
    monitor: str
    monitor_mode: str
    check_val_every_n_epoch: int
    patience: int

@dataclass
class TrainConfig:
    dir: DirConfig
    dataset: DatasetConfig
    model: ModelConfig
    scheduler: SchedulerConfig
    optimizer: OptimizerConfig
    exp_name: str
    folds: int
    n_repeats: int
    phase: str
    target_col: str
    loss: str
    alpha_1: float
    alpha_2: float
    offline: bool
    repeat_interleave: bool

@dataclass
class InferenceConfig:
    dir: DirConfig
    dataset: DatasetConfig
    model: ModelConfig
    exp_name: str
    folds: int
    n_repeats: int
    phase: str
    target_col: str
    use_amp: bool

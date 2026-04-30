# api/config/optimization_config.py

from typing import List, Literal

import yaml
from pydantic import BaseModel, Field


class ObjectiveConfig(BaseModel):
    """Configuração da função objetivo."""
    k_target: int = Field(2, ge=1)
    mae_weight: float = Field(0.5, ge=0.0)
    bias_weight: float = Field(0.3, ge=0.0)
    max_p90_k3: float = Field(1.0, ge=0.0)


class RangeConfig(BaseModel):
    """Range contínuo [low, high]."""
    low: float
    high: float


class SearchSpaceConfig(BaseModel):
    """Espaço de busca dos hiperparâmetros."""
    alpha_daily: RangeConfig = RangeConfig(low=0.05, high=0.6)
    lr: RangeConfig = RangeConfig(low=3e-4, high=3e-3)
    hidden_dim: List[int] = [32, 64, 96]
    conv_filters: List[int] = [16, 32, 48]
    dropout: RangeConfig = RangeConfig(low=0.0, high=0.25)


class OptunaEarlyStoppingConfig(BaseModel):
    """Early stopping agressivo para trials Optuna."""
    monitor: str = "train_loss"
    patience: int = Field(3, ge=1)
    min_delta: float = Field(1e-4, ge=0)


class OptunaCheckpointConfig(BaseModel):
    """Checkpoint durante Optuna (normalmente desabilitado)."""
    save_top_k: int = Field(0, ge=0)
    monitor: str = "val_loss"
    mode: Literal["min", "max"] = "min"


class OptunaTrainerConfig(BaseModel):
    """Configuração do Trainer para trials Optuna (modo fast)."""
    max_epochs: int = Field(25, ge=1)
    accelerator: str = "auto"
    precision: str = "16-mixed"
    gradient_clip_val: float = Field(1.0, ge=0)
    early_stopping: OptunaEarlyStoppingConfig = OptunaEarlyStoppingConfig()
    checkpoint: OptunaCheckpointConfig = OptunaCheckpointConfig()


class OptunaDataConfig(BaseModel):
    """Configuração do DataModule para Optuna (batch maior)."""
    batch_size: int = Field(128, ge=1)
    num_workers: int = Field(2, ge=0)


class OptimizationConfig(BaseModel):
    """Configuração completa da otimização de hiperparâmetros."""
    trainer: OptunaTrainerConfig = OptunaTrainerConfig()
    data: OptunaDataConfig = OptunaDataConfig()
    n_trials: int = Field(30, ge=1)
    max_epochs_per_trial: int = Field(25, ge=1)
    seed: int = 13
    objective: ObjectiveConfig = ObjectiveConfig()
    search_space: SearchSpaceConfig = SearchSpaceConfig()

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "OptimizationConfig":
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

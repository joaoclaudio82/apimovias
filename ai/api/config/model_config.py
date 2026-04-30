# api/config/model_config.py

from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field


class ModelHparams(BaseModel):
    """Hiperparâmetros dos modelos de forecasting."""
    hidden_dim: int = Field(64, ge=1)
    conv_filters: int = Field(32, ge=1)
    conv_kernel: int = Field(3, ge=1)
    conv_layers: int = Field(1, ge=1)
    dropout: float = Field(0.1, ge=0.0, le=1.0)
    loss_heads_type: Literal["huber", "mse", "mae"] = "huber"
    loss_daily_type: Literal["huber", "mse", "mae"] = "mae"
    alpha_daily: float = Field(0.3, ge=0.0)


class OptimizerConfig(BaseModel):
    """Configuração do otimizador AdamW."""
    lr: float = Field(1e-3, gt=0)
    weight_decay: float = Field(1e-4, ge=0)


class ModelConfig(BaseModel):
    """Configuração do modelo de previsão (hiperparâmetros + otimizador)."""
    model: ModelHparams = ModelHparams()
    optimizer: OptimizerConfig = OptimizerConfig()

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ModelConfig":
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls(**data)

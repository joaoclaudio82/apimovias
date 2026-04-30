# api/config/training_config.py

from typing import Dict, List, Literal

import yaml
from pydantic import BaseModel, Field


class NormalizationConfig(BaseModel):
    """Parâmetros do ProfileDatasetNormalizer."""
    cv_max: float = Field(5.0, gt=0)
    ratio_max: float = Field(5.0, gt=0)


class SplitConfig(BaseModel):
    """Configuração de split temporal treino/val/teste."""
    val_size: float = Field(0.15, ge=0.0, le=1.0)
    test_size: float = Field(0.15, ge=0.0, le=1.0)
    shuffle: bool = True
    random_state: int = 42


class DataConfig(BaseModel):
    """Configuração do DataModule."""
    batch_size: int = Field(64, ge=1)
    num_workers: int = Field(0, ge=0)


class EarlyStoppingConfig(BaseModel):
    """Configuração de early stopping."""
    monitor: str = "val_loss"
    patience: int = Field(20, ge=1)
    min_delta: float = Field(1e-4, ge=0)


class CheckpointConfig(BaseModel):
    """Configuração do ModelCheckpoint."""
    monitor: str = "val_loss"
    save_top_k: int = Field(1, ge=1)
    mode: Literal["min", "max"] = "min"


class TrainerConfig(BaseModel):
    """Configuração do PyTorch Lightning Trainer."""
    max_epochs: int = Field(200, ge=1)
    accelerator: str = "auto"
    precision: int = 32
    gradient_clip_val: float = Field(1.0, ge=0)
    early_stopping: EarlyStoppingConfig = EarlyStoppingConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()


class OnnxConfig(BaseModel):
    """Configuração de exportação ONNX."""
    enabled: bool = True
    opset_version: int = Field(17, ge=7)
    verify: bool = True


class TrainingConfig(BaseModel):
    """Configuração de treinamento (normalização + split + trainer + ONNX)."""
    normalization: NormalizationConfig = NormalizationConfig()
    split: SplitConfig = SplitConfig()
    seed: int = 42
    data: DataConfig = DataConfig()
    trainer: TrainerConfig = TrainerConfig()
    onnx: OnnxConfig = OnnxConfig()

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainingConfig":
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls(**data)

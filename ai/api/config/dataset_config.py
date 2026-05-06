# api/config/dataset_config.py

from typing import List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, model_validator


class DatasetConfig(BaseModel):
    """Parâmetros do ProfileDatasetGenerator / CachedDatasetGenerator."""
    min_weeks_general: int = Field(12, ge=1)
    num_weeks_recent: int = Field(4, ge=1)
    horizon_weeks: Union[int, List[int]] = 4
    daily_horizon: int = Field(7, ge=1)
    min_recent_active_days: int = Field(5, ge=0)
    cluster_features: Optional[Literal["prediction", "probabilities"]] = None

    @model_validator(mode="after")
    def _validate_uniform_blocks(self):
        """Garante que horizon_weeks tem blocos de tamanho uniforme."""
        hw = self.horizon_weeks
        if isinstance(hw, list) and len(set(hw)) > 1:
            raise ValueError(
                f"horizon_weeks deve ter blocos de mesmo tamanho, "
                f"recebido: {hw}. Exemplos válidos: 4, [1,1,1,1], [2,2]."
            )
        return self

    @property
    def n_heads(self) -> int:
        """Número de heads (blocos de horizonte)."""
        if isinstance(self.horizon_weeks, int):
            return self.horizon_weeks
        return len(self.horizon_weeks)

    @property
    def block_weeks(self) -> int:
        """Tamanho de cada bloco em semanas."""
        if isinstance(self.horizon_weeks, int):
            return 1
        return self.horizon_weeks[0]

    @property
    def block_days(self) -> int:
        """Tamanho de cada bloco em dias."""
        return self.block_weeks * 7

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "DatasetConfig":
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls(**data)

# api/config/dataset_config.py

from typing import List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    """Parâmetros do ProfileDatasetGenerator / CachedDatasetGenerator."""
    min_weeks_general: int = Field(12, ge=1)
    num_weeks_recent: int = Field(4, ge=1)
    horizon_weeks: Union[int, List[int]] = 4
    daily_horizon: int = Field(7, ge=1)
    min_recent_active_days: int = Field(5, ge=0)
    cluster_features: Optional[Literal["prediction", "probabilities"]] = None

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "DatasetConfig":
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls(**data)

# api/config/data_quality_config.py

from typing import Dict

import yaml
from pydantic import BaseModel, Field


class ThresholdsConfig(BaseModel):
    """Limiares do SeriesQualityFilter para detecção de séries não-modeláveis."""
    min_weeks: int = Field(5, ge=1)
    max_gap: int = Field(3, ge=1)


class DataQualityConfig(BaseModel):
    """Configuração de qualidade de dados: formatação + quantis."""
    thresholds: ThresholdsConfig = ThresholdsConfig()
    quantiles: Dict[str, float] = {
        "prop_km_high": 0.90,
        "prop_h_low": 0.05,
        "corr_low": 0.10,
        "taxa_dias_low": 0.05,
        "gap_medio_high": 0.95,
        "cv_gaps_high": 0.95,
    }

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "DataQualityConfig":
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

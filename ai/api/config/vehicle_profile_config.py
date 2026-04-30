# api/config/vehicle_profile_config.py

from typing import List, Optional

import yaml
from pydantic import BaseModel, Field


class VehicleProfileFeaturesConfig(BaseModel):
    """Features por extractor. None = extractor não utilizado."""
    seg: Optional[List[str]] = ["gap_medio", "cv_gaps", "taxa_dias_ativos", "p75", "iqr"]
    day: Optional[List[str]] = ["mean", "std", "p25", "p75", "iqr", "prob_active", "cv"]
    phase: Optional[List[str]] = ["prob_active", "mean", "p75", "iqr", "cv"]
    cycle: Optional[List[str]] = ["prob_active", "mean", "ratio_fim_inicio"]


class ProfileParamsConfig(BaseModel):
    """Parâmetros do VehicleProfile."""
    sample_size: int = Field(364, ge=1)
    p_upper: int = Field(95, ge=1, le=100)
    n_jobs: int = Field(4, ge=1)
    batch_size: int = Field(10, ge=1)


class VehicleProfileMetricFeaturesConfig(BaseModel):
    """Features por métrica."""
    km: VehicleProfileFeaturesConfig = VehicleProfileFeaturesConfig()
    h: VehicleProfileFeaturesConfig = VehicleProfileFeaturesConfig()


class VehicleProfileConfig(BaseModel):
    """Configuração completa do VehicleProfile."""
    features: VehicleProfileMetricFeaturesConfig = VehicleProfileMetricFeaturesConfig()
    profile: ProfileParamsConfig = ProfileParamsConfig()

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "VehicleProfileConfig":
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls(**data)

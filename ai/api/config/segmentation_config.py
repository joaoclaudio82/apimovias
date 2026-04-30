# api/config/segmentation_config.py

from typing import List, Optional, Tuple

import yaml
from pydantic import BaseModel, Field

class MetricFeaturesConfig(BaseModel):
    """Features por extractor para uma métrica."""
    seg: Optional[List[str]] = ["gap_medio", "cv_gaps", "taxa_dias_ativos", "p75", "iqr"]


class FeaturesConfig(BaseModel):
    """Features para cada extractor."""
    type: List[str] = ["razao_km_h", "proporcao_km", "corr_km_h"]
    km: MetricFeaturesConfig = MetricFeaturesConfig()
    h: MetricFeaturesConfig = MetricFeaturesConfig()


class ClusteringConfig(BaseModel):
    """Parâmetros de clustering."""
    k: Optional[int] = None
    k_range: Tuple[int, int] = (2, 10)


class Stage1ClassificationConfig(BaseModel):
    """Parâmetros de classificação da etapa 1."""
    test_size: float = Field(0.3, gt=0, lt=1)
    percentile_clean: float = Field(0.10, ge=0, le=1)


class UncertaintyConfig(BaseModel):
    """Limiares de incerteza para separação KM/H."""
    low_pct: float = Field(0.05, gt=0, lt=1)
    high_pct: float = Field(0.95, gt=0, lt=1)


class Stage1Config(BaseModel):
    """Configuração da etapa 1: métrica predominante."""
    clustering: ClusteringConfig = ClusteringConfig(k=2, k_range=(2, 10))
    classification: Stage1ClassificationConfig = Stage1ClassificationConfig()
    uncertainty: UncertaintyConfig = UncertaintyConfig()


class Stage2ClassificationConfig(BaseModel):
    """Parâmetros de classificação da etapa 2."""
    test_size: float = Field(0.3, gt=0, lt=1)


class Stage2Config(BaseModel):
    """Configuração da etapa 2: segmentação por métrica."""
    clustering: ClusteringConfig = ClusteringConfig(k=None, k_range=(2, 12))
    classification: Stage2ClassificationConfig = Stage2ClassificationConfig()


class ReportConfig(BaseModel):
    """Configuração do relatório PDF."""
    enabled: bool = True


class SegmentationConfig(BaseModel):
    """Configuração completa do pipeline de segmentação."""
    features: FeaturesConfig = FeaturesConfig()
    stage1: Stage1Config = Stage1Config()
    stage2: Stage2Config = Stage2Config()
    report: ReportConfig = ReportConfig()

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "SegmentationConfig":
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls(**data)

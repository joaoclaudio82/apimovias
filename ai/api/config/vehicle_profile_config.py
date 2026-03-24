# api/config/vehicle_profile_config.py

from pydantic import BaseModel, Field
from typing import Dict
import yaml
import logging

from api.config.path_resolver import resolve_path, resolve_csv_path

logger = logging.getLogger(__name__)


class ProfileConfig(BaseModel):
    """Configuração de geração de perfis"""
    sample_size: int = Field(364, ge=1, description="Tamanho da amostra de histórico")
    p_upper: int = Field(99, ge=1, le=100, description="Percentil para upper bound")
    n_jobs: int = Field(1, description="Paralelização")

class ClassificationModelsConfig(BaseModel):
    """Configuração de modelos de classificação"""
    stage1_type: str
    stage2_displacement: str
    stage2_machine: str


class ClassificationConfig(BaseModel):
    """Configuração de classificação"""
    models: ClassificationModelsConfig
    min_days: int = Field(28, ge=1)


class MappingConfig(BaseModel):
    """Mapeamento de classes"""
    category: Dict[int, str]
    segments: Dict[str, int]


class VehicleProfileConfig(BaseModel):
    """Configuração completa de perfis de veículos"""
    input_data: str = Field(..., description="Path do arquivo de importação de dados")
    profile: ProfileConfig
    classification: ClassificationConfig
    mapping: MappingConfig
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'VehicleProfileConfig':
        """Carrega configuração de arquivo YAML"""
        logger.info(f"Carregando configuração de perfis: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        if "input_data" in data:
            data["input_data"] = resolve_csv_path(data["input_data"], yaml_path)

        classification = data.get("classification", {})
        models = classification.get("models", {})
        for key in ("stage1_type", "stage2_displacement", "stage2_machine"):
            if key in models:
                models[key] = resolve_path(models[key], yaml_path)
        
        config = cls(**data)
        logger.info("Configuração de perfis carregada")
        
        return config
    
    def get_category_from_prediction(self, prediction: int) -> str:
        """Converte predição para category"""
        return self.mapping.category.get(prediction, 'unknown')

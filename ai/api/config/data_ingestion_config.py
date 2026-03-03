# api/config/data_ingestion_config.py

from pydantic import BaseModel, Field
from typing import List, Literal
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)


class FormatterConfig(BaseModel):
    """Configuração do DatasetFormatter"""
    min_days: int = Field(7, ge=1, description="Mínimo de dias ativos")
    min_weeks: int = Field(5, ge=1, description="Mínimo de semanas ativas")
    max_gap: int = Field(3, ge=0, description="Gap máximo entre atividades")


class GeneratorConfig(BaseModel):
    """Configuração do DatasetGenerator"""
    history_size: int = Field(28, ge=1, description="Tamanho do histórico")
    forecast_horizon: int = Field(7, ge=1, description="Horizonte de previsão")
    n_windows_test: int = Field(4, ge=1, description="Número de janelas de teste")
    n_windows_val: int = Field(4, ge=1, description="Número de janelas de validação")
    skip_n_days: int = Field(0, ge=0, description="Dias a pular entre janelas")
    use_effective_period: bool = Field(True, description="Usar período efetivo")


class DataIngestionConfig(BaseModel):
    """Configuração completa de ingestão de dados"""
    dataset_path: str = Field(..., description="Path do dataset de telemetria")
    output_dir: str = Field(..., description="Diretório base de saída")
    formatter: FormatterConfig
    generator: GeneratorConfig
    categories: List[Literal['km_dia_clean', 'h_dia_clean']]
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'DataIngestionConfig':
        """Carrega configuração de arquivo YAML"""
        logger.info(f"Carregando configuração de ingestão: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        config = cls(**data)
        logger.info(f"Configuração carregada: {len(config.categories)} categorias")
        
        return config
    
    def get_output_dir_for_category(self, category: str) -> Path:
        """Retorna diretório de saída para uma categoria"""
        metric = category.split('_')[0]
        return Path(self.output_dir) / metric
    
    def get_segments_dir_for_category(self, category: str) -> Path:
        """Retorna diretório de segmentos para uma categoria"""
        return self.get_output_dir_for_category(category) / 'segments'
    
    def get_windows_dir(self) -> Path:
        """Retorna diretório de janelas (windows)"""
        return Path(self.output_dir) / 'windows'
    
    def get_profile_dir(self) -> Path:
        """Retorna diretório de perfis"""
        return Path(self.output_dir) / 'profile'
# api/config/training_config.py

from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)


class PyTorchModelsConfig(BaseModel):
    """Configuração de modelos PyTorch"""
    enabled: bool = Field(True, description="Se False, pula modelos PyTorch")
    models: List[str] = Field(
        ['DLinear', 'NLinear', 'TSMixerModel', 'TCN', 'NBEATS', 'NHiTS'],
        description="Lista de modelos PyTorch a treinar"
    )
    likelihood: bool = Field(False, description="Usar QuantileRegression")
    incremental: bool = Field(False, description="Treinamento incremental")
    force_reset: bool = Field(True, description="Forçar reset do modelo")
    load_best: bool = Field(True, description="Carregar melhor checkpoint")
    n_epochs: int = Field(100, ge=1, description="Número de épocas")
    batch_size: int = Field(64, ge=1, description="Tamanho do batch")


class GlobalModelsConfig(BaseModel):
    """Configuração de modelos Globais (não-PyTorch)"""
    enabled: bool = Field(True, description="Se False, pula modelos Globais")
    models: List[str] = Field(
        ['LightGBM', 'XGBoost', 'LinearRegression'],
        description="Lista de modelos Globais a treinar"
    )
    stride: int = Field(7, ge=1, description="Stride para janelas")


class TrainingOptionsConfig(BaseModel):
    """Configuração de opções de treinamento"""
    verbose_training: bool = Field(True, description="Logs detalhados")
    save_models: bool = Field(True, description="Salvar modelos")
    save_results: bool = Field(True, description="Salvar resultados")


class TrainingConfig(BaseModel):
    """Configuração completa de treinamento"""
    windows_dir: str = Field(..., description="Diretório com janelas")
    profile_dir: str = Field(..., description="Diretório de perfis")
    work_dir: str = Field(..., description="Diretório de trabalho")
    cache_dir: str = Field(..., description="Diretório de cache")
    seed: int = Field(13, description="Seed para reprodutibilidade")
    eval_metric: Literal['wmape', 'mape', 'rmse', 'mae'] = 'wmape'
    
    # ✅ Parsear manualmente
    pytorch: PyTorchModelsConfig
    global_: GlobalModelsConfig = Field(alias='global')
    options: TrainingOptionsConfig
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'TrainingConfig':
        """Carrega configuração de arquivo YAML"""
        logger.info(f"Carregando configuração de treinamento: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # ✅ Reorganizar estrutura do YAML
        if 'training' in data:
            training = data.pop('training')
            
            # Extrair models
            if 'models' in training:
                models = training.pop('models')
                data['pytorch'] = models.get('pytorch', {})
                data['global'] = models.get('global', {})
            
            # Extrair options
            data['options'] = {
                'verbose_training': training.get('verbose_training', True),
                'save_models': training.get('save_models', True),
                'save_results': training.get('save_results', True)
            }
        
        config = cls(**data)
        logger.info(f"Configuração carregada: seed={config.seed}")
        
        return config
    
    def get_pytorch_models_to_train(self) -> Optional[List[str]]:
        """Retorna lista de modelos PyTorch a treinar"""
        if not self.pytorch.enabled:
            return None
        return self.pytorch.models
    
    def get_global_models_to_train(self) -> Optional[List[str]]:
        """Retorna lista de modelos Globais a treinar"""
        if not self.global_.enabled:
            return None
        return self.global_.models
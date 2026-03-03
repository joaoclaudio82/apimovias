# api/config/predictor_config.py

from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional, Literal
from pathlib import Path
import yaml
import logging

from api.config.path_resolver import resolve_path

logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    """Configuração de um modelo específico"""
    model_name: str = Field(..., description="Nome do modelo (ex: DLinearModel)")
    weight: float = Field(1.0, ge=0.0, le=1.0, description="Peso no ensemble")


class SegmentConfig(BaseModel):
    """Configuração para um segmento específico"""
    segment: int = Field(..., ge=0, description="Número do segmento")
    models: List[ModelConfig] = Field(..., description="Lista de modelos (ensemble se > 1)")
    
    @field_validator('models')
    @classmethod
    def validate_weights(cls, models: List[ModelConfig]) -> List[ModelConfig]:
        """Valida que pesos somam 1.0"""
        total_weight = sum(m.weight for m in models)
        if not abs(total_weight - 1.0) < 1e-6:
            raise ValueError(f"Soma dos pesos deve ser 1.0, recebido: {total_weight}")
        return models


class CategoryConfig(BaseModel):
    """Configuração para uma categoria (km ou h)"""
    category: Literal['km', 'h']
    segments: List[SegmentConfig] = Field(..., description="Configuração por segmento")


class PredictorConfig(BaseModel):
    """Configuração geral do sistema de predição"""
    models_root_dir: str = Field(..., description="Diretório raiz dos modelos")
    history_size: int = Field(28, ge=1, description="Tamanho do histórico (global)")
    forecast_horizon: int = Field(7, ge=1, description="Horizonte de previsão (global)")
    max_steps: int = Field(365, ge=1, description="Máximo de steps para predição (global)")
    categories: List[CategoryConfig] = Field(..., description="Configurações por categoria")
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'PredictorConfig':
        """Carrega configuração de arquivo YAML"""
        logger.info(f"Carregando configuração de: {yaml_path}")

        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        if "models_root_dir" in data:
            data["models_root_dir"] = resolve_path(data["models_root_dir"], yaml_path)

        config = cls(**data)
        logger.info(f"Configuração carregada: {len(config.categories)} categorias")

        return config
    
    def get_segment_config(self, category: str, segment: int) -> Optional[SegmentConfig]:
        """Obtém configuração de um segmento específico"""
        for cat_config in self.categories:
            if cat_config.category == category:
                for seg_config in cat_config.segments:
                    if seg_config.segment == segment:
                        return seg_config
        return None
    
    def find_model_path(self, model_name: str, category: str, segment: int) -> Optional[Path]:
        """
        Encontra path do modelo no diretório raiz
        
        Procura por padrão: {model_name}/dataset_{category}_*_class{segment}_*
        
        Examples
        --------
        >>> config.find_model_path('DLinearModel', 'km', 2)
        Path('models/DLinearModel/dataset_km_dia_clean_..._class2_...')
        """
        models_root = Path(self.models_root_dir)
        model_dir = models_root / model_name
        
        if not model_dir.exists():
            logger.warning(f"Diretório do modelo não encontrado: {model_dir}")
            return None
        
        # Procurar subdiretório que corresponde ao padrão
        pattern = f"dataset_{category}_*_class{segment}_*"
        
        matches = list(model_dir.glob(pattern))
        
        if not matches:
            logger.warning(
                f"Nenhum modelo encontrado para padrão: {model_dir}/{pattern}"
            )
            return None
        
        if len(matches) > 1:
            logger.warning(
                f"Múltiplos modelos encontrados para {pattern}, usando o mais recente"
            )
            matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        logger.info(f"Modelo encontrado: {matches[0]}")
        
        return matches[0]
    
    def validate_all_models_exist(self) -> Dict[str, List[str]]:
        """
        Valida que todos os modelos configurados existem
        
        Returns
        -------
        dict
            {'found': [...], 'missing': [...]}
        """
        found = []
        missing = []
        
        for cat_config in self.categories:
            for seg_config in cat_config.segments:
                for model_config in seg_config.models:
                    model_path = self.find_model_path(
                        model_config.model_name,
                        cat_config.category,
                        seg_config.segment
                    )
                    
                    key = f"{cat_config.category}_{seg_config.segment}_{model_config.model_name}"
                    
                    if model_path and model_path.exists():
                        found.append(key)
                    else:
                        missing.append(key)
        
        return {'found': found, 'missing': missing}

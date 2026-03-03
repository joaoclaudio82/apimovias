# api/schemas/training_schemas.py

from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class TrainingRequest(BaseModel):
    """Requisição de treinamento"""
    train_pytorch: bool = Field(True, description="Treinar modelos PyTorch")
    train_global: bool = Field(True, description="Treinar modelos Globais")
    verbose: bool = Field(True, description="Logs detalhados")


class TrainingResponse(BaseModel):
    """Resposta do treinamento"""
    status: str
    models_trained: List[str]
    work_dir: str
    pytorch_trained: bool
    global_trained: bool
    errors: Optional[Dict[str, str]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "models_trained": ["DLinear", "NLinear", "LightGBM", "XGBoost"],
                "work_dir": "../models_features",
                "pytorch_trained": True,
                "global_trained": True,
                "errors": None
            }
        }
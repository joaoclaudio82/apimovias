# api/config/predictor_config.py

from pathlib import Path

import yaml
from pydantic import BaseModel


class PredictorModelsConfig(BaseModel):
    """Nomes dos ficheiros ONNX por métrica."""
    km: str = "forecasting_moe_km.onnx"
    h: str = "forecasting_moe_h.onnx"


class PredictorConfig(BaseModel):
    """Configuração do predictor: modelos ONNX a utilizar."""
    models: PredictorModelsConfig = PredictorModelsConfig()

    def model_path(self, target: str, forecasting_dir: str) -> Path:
        """Retorna o caminho absoluto do modelo ONNX para o target."""
        filename = getattr(self.models, target)
        return Path(forecasting_dir) / filename

    def validate_all_models_exist(self, forecasting_dir: str) -> dict:
        """Verifica existência dos modelos configurados."""
        found, missing = [], []
        for target in ("km", "h"):
            p = self.model_path(target, forecasting_dir)
            (found if p.exists() else missing).append(str(p))
        return {"found": found, "missing": missing}

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PredictorConfig":
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

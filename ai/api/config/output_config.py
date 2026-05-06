# api/config/output_config.py

from pathlib import Path

import yaml
from pydantic import BaseModel

from api.config.path_resolver import resolve_path


class DataConfig(BaseModel):
    """Diretórios de dados."""
    train_dataset: str
    profiles: str
    cache: str


class LogsConfig(BaseModel):
    """Diretórios de logs e artefactos gerados."""
    segmentation: str
    training: str


class ModelsOutputConfig(BaseModel):
    """Diretórios dos modelos gerados."""
    classification: str
    forecasting: str


class OutputConfig(BaseModel):
    """Configuração centralizada de diretórios de saída."""
    data: DataConfig
    logs: LogsConfig
    models: ModelsOutputConfig

    # -- helpers de paths derivados --

    def train_data_path(self, target: str) -> Path:
        """CSV de treino gerado pelo pipeline de segmentação."""
        return Path(self.data.train_dataset) / f"{target}.csv"

    def profile_dir(self, target: str) -> Path:
        """Diretório do perfil de um target."""
        return Path(self.data.profiles) / target

    def classifier_path(self, target: str, ext: str = "pkl") -> Path:
        """Melhor classificador stage2 de um target."""
        return Path(self.models.classification) / "stage2" / f"stage2_{target}_BEST.{ext}"

    def forecasting_model_path(self, target: str) -> Path:
        """Modelo ONNX de forecasting de um target."""
        return Path(self.models.forecasting) / f"forecasting_model_{target}.onnx"

    def dataset_cache_dir(self, target: str) -> Path:
        """Diretório de cache do dataset de um target."""
        return Path(self.data.cache) / target

    def training_dir(self, target: str) -> Path:
        """Diretório de logs e relatório de treinamento de um target."""
        return Path(self.logs.training) / target
    
    def predictions_path(self, target: str) -> Path:
        """Diretório de predições de um target."""
        return Path(self.logs.training) / target / "predictions"

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "OutputConfig":
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Resolve root relative to the YAML file, then use it as base for all paths
        root = Path(resolve_path(data.pop("root", "."), yaml_path))

        def _resolve(value: str) -> str:
            p = Path(value)
            if p.is_absolute():
                return str(p)
            return str((root / p).resolve())

        for section in ("data", "logs", "models"):
            if section in data:
                for key in data[section]:
                    data[section][key] = _resolve(data[section][key])

        return cls(**data)

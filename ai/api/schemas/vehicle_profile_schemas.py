# api/schemas/vehicle_profile_schemas.py

from pydantic import BaseModel
from typing import Dict, Any


class ProfileUpdatePathRequest(BaseModel):
    """Requisição para atualização via CSV compartilhado."""
    target: str
    csv_path: str


class ProfileImportResponse(BaseModel):
    """Resposta da importação de perfis"""
    target: str
    category: str
    n_vehicles: int
    message: str = "Importação concluída com sucesso"


class ProfileUpdateResponse(BaseModel):
    """Resposta da atualização de perfis"""
    target: str
    n_vehicles: int
    message: str = "Atualização concluída com sucesso"


class VehicleInfoResponse(BaseModel):
    """Informações básicas do veículo"""
    id: int
    category: str
    segment: int
    first_activity_date: str
    last_activity_date: str
    upper: float
    n_samples: int


class StatisticsResponse(BaseModel):
    """Estatísticas agregadas por categoria"""
    status: str
    statistics: Dict[str, Dict[str, Any]]

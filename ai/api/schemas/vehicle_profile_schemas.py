# api/schemas/vehicle_profile_schemas.py

from pydantic import BaseModel
from typing import Dict, Any


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

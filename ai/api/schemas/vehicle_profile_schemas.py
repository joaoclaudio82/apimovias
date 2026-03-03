# api/schemas/vehicle_profile_schemas.py

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List


class ImportFromFileRequest(BaseModel):
    """Requisição para importar perfis usando fonte configurada"""
    vehicle_ids: Optional[List[int]] = Field(None, description="IDs específicos para processar")


class ImportFromFileResponse(BaseModel):
    """Resposta da importação"""
    status: str
    total_vehicles: int
    existing_updated: int
    new_added: int
    km_vehicles: int
    h_vehicles: int
    deleted_vehicles: int


class VehicleInfoResponse(BaseModel):
    """Informações básicas do veículo"""
    id: int
    category: str
    segment: int
    samples_start_date: str
    samples_end_date: str
    upper: float
    n_samples: int


class StatisticsResponse(BaseModel):
    """Estatísticas agregadas por categoria"""
    status: str
    statistics: Dict[str, Dict[str, Any]]

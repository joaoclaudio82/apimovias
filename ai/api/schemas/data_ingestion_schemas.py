# api/schemas/data_ingestion_schemas.py

from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class DataIngestionRequest(BaseModel):
    """Requisição de ingestão de dados"""
    clean_intermediate: bool = Field(
        False,
        description="Se True, remove arquivos intermediários (mantém só windows)"
    )
    verbose: bool = Field(
        True,
        description="Se True, exibe logs detalhados"
    )


class DataIngestionResponse(BaseModel):
    """Resposta da ingestão de dados"""
    status: str
    categories_processed: List[str]
    total_windows_generated: int
    output_dir: str
    profile_dir: str  
    intermediate_cleaned: bool
    details: Optional[Dict] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "categories_processed": ["km_dia_clean", "h_dia_clean"],
                "total_windows_generated": 12,
                "output_dir": "../datasets/windows",
                "profile_dir": "../datasets/profile",
                "intermediate_cleaned": True
            }
        }
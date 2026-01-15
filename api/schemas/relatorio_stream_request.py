from datetime import date
from typing import Optional
from pydantic import BaseModel, Field

class RelatorioStreamRequest(BaseModel):
    id_start: int = Field(...)
    id_end: int = Field(...)
    data_ini: Optional[date] = None
    data_fim: Optional[date] = None

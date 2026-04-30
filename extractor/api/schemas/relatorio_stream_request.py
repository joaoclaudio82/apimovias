from datetime import date
from typing import Optional
from pydantic import BaseModel, Field

class RelatorioStreamRequest(BaseModel):
    id_start: Optional[int] = Field(None)
    id_end: Optional[int] = Field(None)
    data_ini: Optional[date] = None
    data_fim: Optional[date] = None

from datetime import date
from typing import Optional
from pydantic import BaseModel, Field

class RelatorioRequest(BaseModel):
    veiculo_id: int = Field(...)
    data_ini: Optional[date] = None
    data_fim: Optional[date] = None

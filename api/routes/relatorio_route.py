from fastapi import APIRouter, HTTPException, status
from api.schemas.relatorio_request import RelatorioRequest
from api.schemas.relatorio_stream_request import RelatorioStreamRequest
from api.services.relatorio_service import RelatorioService

router = APIRouter(prefix="/relatorios", tags=["Relatório"])

@router.get("/download")
def download_csv():
    if not (file := RelatorioService.download()):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    return file

@router.post("/", status_code=status.HTTP_204_NO_CONTENT)
def create_csv(payload: RelatorioRequest):
    RelatorioService.create_and_append_csv(
        id_start=payload.veiculo_id,
        id_end=payload.veiculo_id,
        data_ini=payload.data_ini,
        data_fim=payload.data_fim
    )

@router.post("/batch", status_code=status.HTTP_204_NO_CONTENT)
def create_csv_batch(payload: RelatorioStreamRequest):
    RelatorioService.create_and_append_csv(
        id_start=payload.id_start,
        id_end=payload.id_end,
        data_ini=payload.data_ini,
        data_fim=payload.data_fim
    )

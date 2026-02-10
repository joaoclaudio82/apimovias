import logging
import time
from fastapi import APIRouter, HTTPException, status
from api.schemas.relatorio_request import RelatorioRequest
from api.schemas.relatorio_stream_request import RelatorioStreamRequest
from api.services.relatorio_service import RelatorioService

router = APIRouter(prefix="/relatorios", tags=["Relatório"])
logger = logging.getLogger(__name__)

@router.get("/download")
def download_csv():
    if not (file := RelatorioService.download()):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    return file

@router.post("/", status_code=status.HTTP_204_NO_CONTENT)
def create_csv(payload: RelatorioRequest):
    start = time.perf_counter()
    ids = [payload.veiculo_id]
    try:
        RelatorioService.create_and_append_csv(
            id_start=payload.veiculo_id,
            id_end=payload.veiculo_id,
            data_ini=payload.data_ini,
            data_fim=payload.data_fim
        )
    except Exception:
        elapsed = time.perf_counter() - start
        logger.exception(
            "create_csv failed | veiculos=%d | ids=%s | elapsed_ms=%.2f",
            len(ids),
            ids,
            elapsed * 1000.0,
        )
        raise
    else:
        elapsed = time.perf_counter() - start
        logger.info(
            "create_csv ok | veiculos=%d | ids=%s | elapsed_ms=%.2f",
            len(ids),
            ids,
            elapsed * 1000.0,
        )

@router.post("/batch", status_code=status.HTTP_204_NO_CONTENT)
def create_csv_batch(payload: RelatorioStreamRequest):
    start = time.perf_counter()
    ids = list(range(payload.id_start, payload.id_end + 1))
    try:
        RelatorioService.create_and_append_csv(
            id_start=payload.id_start,
            id_end=payload.id_end,
            data_ini=payload.data_ini,
            data_fim=payload.data_fim
        )
    except Exception:
        elapsed = time.perf_counter() - start
        logger.exception(
            "create_csv_batch failed | veiculos=%d | ids=%s | elapsed_ms=%.2f",
            len(ids),
            ids,
            elapsed * 1000.0,
        )
        raise
    else:
        elapsed = time.perf_counter() - start
        logger.info(
            "create_csv_batch ok | veiculos=%d | ids=%s | elapsed_ms=%.2f",
            len(ids),
            ids,
            elapsed * 1000.0,
        )

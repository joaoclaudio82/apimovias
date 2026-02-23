import logging
import time
from fastapi import APIRouter, HTTPException, status
from api.schemas.relatorio_request import RelatorioRequest
from api.services.ai_integration_service import AiIntegrationService
from api.services.relatorio_service import RelatorioService
from api.settings import settings

router = APIRouter(prefix="/relatorios", tags=["Relatório"])
logger = logging.getLogger(__name__)


def _sync_ai_if_enabled() -> None:
    if not settings.AI_INTEGRATION_ENABLED:
        logger.info('Integração AI desabilitada (AI_INTEGRATION_ENABLED=false)')
        return

    start = time.perf_counter()
    try:
        result = AiIntegrationService.sync_profiles_from_csv()
    except Exception as exc:
        elapsed = time.perf_counter() - start
        logger.exception(
            'sync_ai failed | elapsed_ms=%.2f | error=%s',
            elapsed * 1000.0,
            str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f'Falha na sincronização com módulo AI: {exc}',
        ) from exc

    elapsed = time.perf_counter() - start
    logger.info(
        'sync_ai ok | elapsed_ms=%.2f | result=%s',
        elapsed * 1000.0,
        result,
    )


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
        _sync_ai_if_enabled()
        elapsed = time.perf_counter() - start
        logger.info(
            "create_csv ok | veiculos=%d | ids=%s | elapsed_ms=%.2f",
            len(ids),
            ids,
            elapsed * 1000.0,
        )

@router.post("/batch", status_code=status.HTTP_204_NO_CONTENT)
def create_csv_batch():
    start = time.perf_counter()
    try:
        RelatorioService.create_and_append_csv_all()
    except Exception:
        elapsed = time.perf_counter() - start
        logger.exception(
            "create_csv_batch failed | mode=all_veiculos_all_dias | elapsed_ms=%.2f",
            elapsed * 1000.0,
        )
        raise
    else:
        _sync_ai_if_enabled()
        elapsed = time.perf_counter() - start
        logger.info(
            "create_csv_batch ok | mode=all_veiculos_all_dias | elapsed_ms=%.2f",
            elapsed * 1000.0,
        )


@router.post("/sync-ai")
def sync_ai_profiles():
    if not settings.AI_INTEGRATION_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail='Integração AI está desabilitada. Ajuste AI_INTEGRATION_ENABLED=true.',
        )

    start = time.perf_counter()
    try:
        result = AiIntegrationService.sync_profiles_from_csv()
    except Exception as exc:
        elapsed = time.perf_counter() - start
        logger.exception(
            'sync_ai_profiles failed | elapsed_ms=%.2f | error=%s',
            elapsed * 1000.0,
            str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f'Falha na sincronização com módulo AI: {exc}',
        ) from exc

    elapsed = time.perf_counter() - start
    logger.info('sync_ai_profiles ok | elapsed_ms=%.2f', elapsed * 1000.0)
    return {'status': 'ok', **result}

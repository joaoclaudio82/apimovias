# api/app.py

import asyncio
from fastapi import FastAPI
from contextlib import asynccontextmanager, suppress
import logging
from pathlib import Path
from sqlalchemy import select, func

from api.database import engine, session_context
from api.models import table_registry, Vehicle
from api.routers.vehicle_profiles import router as vehicle_profiles_router
from api.config.prediction_config import PredictorConfig
from api.services.vehicle_profile_service import VehicleProfileService

logger = logging.getLogger("uvicorn")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCKER_SHARED_CSV_PATH = Path('/shared/movias.csv')
LOCAL_SHARED_CSV_FALLBACK = PROJECT_ROOT / 'extractor' / 'data' / 'movias.csv'
SYNC_TARGETS = ('km_dia_clean', 'h_dia_clean')
CSV_SYNC_INTERVAL_SECONDS = 30


def _resolve_shared_csv_path() -> Path:
    if DOCKER_SHARED_CSV_PATH.exists():
        return DOCKER_SHARED_CSV_PATH
    if LOCAL_SHARED_CSV_FALLBACK.exists():
        return LOCAL_SHARED_CSV_FALLBACK
    return DOCKER_SHARED_CSV_PATH


async def _profiles_table_is_empty() -> bool:
    async with session_context() as session:
        count = await session.scalar(select(func.count()).select_from(Vehicle))
    return int(count or 0) == 0


async def _sync_profiles_from_shared_csv(app: FastAPI) -> None:
    csv_file = _resolve_shared_csv_path()
    if not csv_file.exists():
        if not getattr(app.state, 'csv_missing_logged', False):
            logger.warning(
                'CSV compartilhado não encontrado | docker=%s | local=%s | resolved=%s',
                str(DOCKER_SHARED_CSV_PATH),
                str(LOCAL_SHARED_CSV_FALLBACK),
                str(csv_file),
            )
            app.state.csv_missing_logged = True
        return

    app.state.csv_missing_logged = False
    current_mtime = csv_file.stat().st_mtime_ns
    last_mtime = getattr(app.state, 'csv_shared_last_mtime', None)
    if last_mtime is not None and current_mtime <= last_mtime:
        return

    targets = SYNC_TARGETS
    if not targets:
        logger.warning('Lista de targets vazia. Sincronização automática ignorada.')
        app.state.csv_shared_last_mtime = current_mtime
        return

    logger.info(
        'Sincronizando CSV compartilhado | path=%s | targets=%s',
        str(csv_file),
        ','.join(targets),
    )
    async with session_context() as session:
        service = VehicleProfileService(session)
        for target in targets:
            try:
                result = await service.update_from_csv_path(
                    target=target,
                    csv_path=str(csv_file),
                )
                logger.info(
                    'Sync CSV automático ok | target=%s | n_vehicles=%s',
                    target,
                    int(result.get('n_vehicles', 0)),
                )
            except Exception as exc:
                logger.exception(
                    'Sync CSV automático falhou | target=%s | error=%s',
                    target,
                    str(exc),
                )

    app.state.csv_shared_last_mtime = current_mtime


async def _csv_sync_worker(app: FastAPI) -> None:
    interval = max(5, int(CSV_SYNC_INTERVAL_SECONDS))
    while True:
        try:
            await _sync_profiles_from_shared_csv(app)
        except Exception as exc:
            logger.exception('Loop de sync CSV falhou: %s', str(exc))
        await asyncio.sleep(interval)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle do app"""
    
    logger.info("Iniciando aplicação...")
    
    # Criar tabelas
    async with engine.begin() as conn:
        await conn.run_sync(table_registry.metadata.create_all)
    logger.info("Tabelas criadas/verificadas")
    
    # Carregar configuração de predição
    try:
        config_path = Path(__file__).parent.parent / "config" / "predictor_config.yaml"
        
        if config_path.exists():
            predictor_config = PredictorConfig.from_yaml(str(config_path))
            
            # ✅ Armazenar no app.state em vez de variável global
            app.state.predictor_config = predictor_config
            
            validation = predictor_config.validate_all_models_exist()
            logger.info(f"✅ Modelos encontrados: {len(validation['found'])}")
            
            if validation['missing']:
                logger.warning(f"⚠️  Modelos faltando: {len(validation['missing'])}")
        else:
            logger.warning(f"⚠️  Arquivo não encontrado: {config_path}")
            app.state.predictor_config = None
    
    except Exception as e:
        logger.error(f"❌ Erro ao carregar configuração: {e}")
        app.state.predictor_config = None

    app.state.csv_sync_task = None
    app.state.csv_shared_last_mtime = None
    app.state.csv_missing_logged = False
    resolved_csv = _resolve_shared_csv_path()
    profiles_empty = await _profiles_table_is_empty()
    if resolved_csv.exists():
        if profiles_empty:
            logger.info(
                'Base de perfis vazia. Sincronização inicial de CSV será executada.',
            )
        else:
            app.state.csv_shared_last_mtime = resolved_csv.stat().st_mtime_ns
            logger.info(
                'CSV compartilhado baseline carregado | path=%s',
                str(resolved_csv),
            )
    logger.info(
        'CSV compartilhado detectado | docker=%s | local=%s | resolved=%s',
        str(DOCKER_SHARED_CSV_PATH),
        str(LOCAL_SHARED_CSV_FALLBACK),
        str(resolved_csv),
    )
    app.state.csv_sync_task = asyncio.create_task(_csv_sync_worker(app))
    logger.info(
        "Sincronização automática de CSV ativada (interval=%ss)",
        int(CSV_SYNC_INTERVAL_SECONDS),
    )

    logger.info(f"🔍 Config carregado? {app.state.predictor_config is not None}")
    logger.info("✅ Aplicação iniciada com sucesso!")
    
    yield
    
    logger.info("Finalizando aplicação...")
    task = getattr(app.state, 'csv_sync_task', None)
    if task is not None:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
    await engine.dispose()
    logger.info("✅ Conexões fechadas.")


app = FastAPI(
    title="Movias AI API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)


@app.get('/health', tags=['health'])
async def health():
    return {'status': 'ok'}


# Incluir routers
app.include_router(vehicle_profiles_router)

try:
    from api.routers.predictions import router as predictions_router
    app.include_router(predictions_router)
except Exception as exc:
    logger.warning(
        'Router de predições desabilitado no startup (%s)',
        str(exc),
    )

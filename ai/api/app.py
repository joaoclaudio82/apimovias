# api/app.py
import os
os.environ.setdefault("MPLBACKEND", "Agg")

from http import HTTPStatus
from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging
from pathlib import Path

from api.database import engine
from api.models import table_registry
from api.schemas import Message
from api.routers.training_pipeline import router as training_pipeline_router
from api.routers.profiling import router as profiling_router
from api.routers.prediction import router as prediction_router

from api.config.training_config import TrainingConfig

logger = logging.getLogger("uvicorn")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("  🔄 Iniciando aplicação...")
        # Criar tabelas
    async with engine.begin() as conn:
        await conn.run_sync(table_registry.metadata.create_all)
    logger.info("  ✅ Banco de dados inicializado")

    """Lifecycle do app"""
    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║              🚛  MOVIAS AI API  v1.0.0                   ║")
    logger.info("║                                                          ║")
    logger.info("║  Análise de perfis de veículos e predição de produção    ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")
    logger.info("")


    logger.info("  ✅ Routers registrados: /profiling, /prediction, /pipeline")
    logger.info("  ✅ Documentação disponível em /docs e /redoc")
    logger.info("")
    logger.info("  🟢 API pronta para receber requisições")
    logger.info("")
    yield
    
    logger.info("")
    logger.info("  🔴 Finalizando aplicação...")
    await engine.dispose()
    logger.info("  ✅ Conexões fechadas")


app = FastAPI(
    title="Movias AI API",
    description="""
API para análise de perfis de veículos e predição de produção.

## Módulos

### 🚗 Profiling
- **Ingestão de dados** — Upload de CSV com atividade diária, execução automática de profiling e predição
- **Veículos** — Listagem de veículos com metadados e qualidade
- **Detalhes** — Perfil de features, metadados e classificação por veículo/target
- **Metadata** — Histórico de perfis carregados

### 📈 Predição
- **Predição** — Execução de modelos ONNX (multihead/MoE) para previsão de H e KM
- **Backtest** — Comparação de predições com valores reais (diário e blocos semanais)

### ⚙️ Pipeline de Treino
- **Segmentação** — Clusterização de veículos
- **Perfis** — Geração de perfis de veículos
- **Datasets** — Preparação de dados de treino
- **Otimização** — Busca de hiperparâmetros e treino de modelos
- **Pipeline completo** — Execução sequencial de todas as etapas
- **Modelos** — Gestão de modelos ONNX ativos
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


@app.get(
    '/health',
    status_code=HTTPStatus.OK,
    response_model=Message,
    tags=['health'],
    summary='Verificar status da API',
)
async def read_root():
    """Retorna status de saúde da API."""
    return {'message': 'Movias AI API - Online'}


app.include_router(training_pipeline_router)
app.include_router(profiling_router)
app.include_router(prediction_router)


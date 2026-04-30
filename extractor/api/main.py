import logging
import os
from fastapi import FastAPI
from api.routes.relatorio_route import router as relatorio_router

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

app = FastAPI(title='Movias Extractor API')
app.include_router(relatorio_router)


@app.get('/health', tags=['health'])
def health():
    return {'status': 'ok'}

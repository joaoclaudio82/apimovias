from fastapi import FastAPI
from api.routes.relatorio_route import router as relatorio_router

app = FastAPI(title='Movias: API')
app.include_router(relatorio_router)

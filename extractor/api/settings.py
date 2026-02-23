from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')

    # Database
    DATABASE_HOST: str
    DATABASE_PORT: int
    DATABASE_DATABASE: str
    DATABASE_USER: str
    DATABASE_PASSWORD: str
    DATABASE_SSL_MODE: str
    RELATORIO_QUERY_BATCH_SIZE: int = 350
    RELATORIO_QUERY_PARALLEL_WORKERS: int = 4

    # Integração com módulo AI
    AI_INTEGRATION_ENABLED: bool = False
    AI_API_BASE_URL: str = 'http://localhost:8010'
    AI_SERVICE_CLIENT_ID: str = 'movias'
    AI_SERVICE_CLIENT_SECRET: str = 'moviaskey'
    AI_SYNC_TARGETS: str = 'km_dia_clean,h_dia_clean'
    AI_SHARED_CSV_PATH: str = '/shared/movias.csv'
    AI_HTTP_TIMEOUT_SECONDS: int = 180

    # Paths
    ROOT_PATH: Path = Path(__file__).resolve().parents[1]
    DATA_PATH: Path = ROOT_PATH / 'data'
    CSV_PATH: Path = DATA_PATH / 'movias.csv'

    def ensure_paths(self) -> None:
        self.DATA_PATH.mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.ensure_paths()

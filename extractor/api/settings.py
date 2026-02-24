from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',
    )

    # Database
    DATABASE_HOST: str
    DATABASE_PORT: int
    DATABASE_DATABASE: str
    DATABASE_USER: str
    DATABASE_PASSWORD: str
    DATABASE_SSL_MODE: str
    RELATORIO_QUERY_BATCH_SIZE: int = 350
    RELATORIO_QUERY_PARALLEL_WORKERS: int = 4
    LOG_LEVEL: str = 'INFO'

    # Paths
    ROOT_PATH: Path = Path(__file__).resolve().parents[1]
    DATA_PATH: Path = ROOT_PATH / 'data'
    CSV_PATH: Path = DATA_PATH / 'movias.csv'

    def ensure_paths(self) -> None:
        self.DATA_PATH.mkdir(parents=True, exist_ok=True)
        self.CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.ensure_paths()

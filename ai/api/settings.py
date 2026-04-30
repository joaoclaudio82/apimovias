from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AliasChoices, Field

# Directório base do pacote ai/ (pai de api/)
_AI_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DB = f"sqlite+aiosqlite:///{_AI_ROOT / 'database.db'}"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',
    )

    DATABASE_URL: str = Field(
        default=_DEFAULT_DB,
        validation_alias=AliasChoices('DATABASE_URL', 'AI_DATABASE_URL'),
    )

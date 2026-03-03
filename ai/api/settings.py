from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AliasChoices, Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',
    )

    DATABASE_URL: str = Field(
        default='sqlite+aiosqlite:///database.db',
        validation_alias=AliasChoices('DATABASE_URL', 'AI_DATABASE_URL'),
    )

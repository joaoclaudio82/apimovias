from __future__ import annotations

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def resolve_path(value: str, yaml_path: str) -> str:
    """
    Resolve path values relative to the YAML file directory.
    """
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)

    base_dir = Path(yaml_path).resolve().parent
    return str((base_dir / path).resolve())


def resolve_csv_path(value: str, yaml_path: str) -> str:
    """
    Resolve CSV path with sensible fallbacks for local and Docker execution.
    """
    configured_path = Path(resolve_path(value, yaml_path))
    if configured_path.exists():
        return str(configured_path)

    env_csv = os.getenv("CSV_PATH")
    if env_csv:
        env_candidate = Path(env_csv).expanduser()
        if not env_candidate.is_absolute():
            env_candidate = (Path(yaml_path).resolve().parent / env_candidate).resolve()
        if env_candidate.exists() or env_candidate.parent.exists():
            logger.warning(
                "CSV configurado não encontrado (%s). Usando CSV_PATH=%s",
                configured_path,
                env_candidate,
            )
            return str(env_candidate)

    shared_csv = Path("/shared/movias.csv")
    if shared_csv.parent.exists():
        logger.warning(
            "CSV configurado não encontrado (%s). Usando fallback Docker: %s",
            configured_path,
            shared_csv,
        )
        return str(shared_csv)

    yaml_resolved = Path(yaml_path).resolve()
    for parent in yaml_resolved.parents:
        local_csv = parent / "extractor" / "data" / "movias.csv"
        if local_csv.parent.exists():
            logger.warning(
                "CSV configurado não encontrado (%s). Usando fallback local: %s",
                configured_path,
                local_csv,
            )
            return str(local_csv)

    return str(configured_path)

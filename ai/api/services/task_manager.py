# api/services/task_manager.py

"""
Gerenciador de tarefas em background para etapas do pipeline.

Garante que apenas uma execução por (step, target) rode simultaneamente.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Dict, Tuple

logger = logging.getLogger(__name__)

# Chave: (step, target) → asyncio.Task
_running_tasks: Dict[Tuple[str, str], asyncio.Task] = {}


def is_running(step: str, target: str) -> bool:
    """Verifica se uma tarefa está em execução para (step, target)."""
    key = (step, target)
    task = _running_tasks.get(key)
    return task is not None and not task.done()


def any_running() -> bool:
    """Verifica se há qualquer tarefa em execução."""
    return any(not task.done() for task in _running_tasks.values())


def submit(
    step: str,
    target: str,
    coro: Awaitable,
) -> asyncio.Task:
    """
    Submete uma coroutine como background task.

    Raises ``RuntimeError`` se já houver tarefa em execução para (step, target).
    """
    if is_running(step, target):
        raise RuntimeError(f"{step}/{target} já está em execução.")

    key = (step, target)

    async def _wrapper():
        try:
            await coro
        except Exception:
            logger.exception("Task %s/%s falhou", step, target)
        finally:
            _running_tasks.pop(key, None)

    task = asyncio.create_task(_wrapper())
    _running_tasks[key] = task
    return task

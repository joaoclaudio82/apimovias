"""
Versionamento de artefactos do pipeline via cadeia de manifests.

Cada etapa (segmentação → perfis → dataset → treinamento) escreve um
``manifest.json`` no seu diretório de saída.  O manifest contém:

- ``stage``: nome da etapa
- ``created_at``: timestamp UTC
- ``config_hash``: SHA-256 da configuração usada
- ``output_hash``: SHA-256 dos ficheiros de saída chave
- ``parent``: ``{"stage": ..., "output_hash": ...}`` da etapa anterior

Antes de executar, cada etapa pode verificar que o ``output_hash``
actual do pai coincide com o registado no seu manifest anterior,
garantindo consistência entre etapas.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest.json"


class StaleArtifactError(RuntimeError):
    """Os artefactos de uma etapa anterior foram alterados desde a última execução."""


class PipelineManifest:
    """Leitura e escrita do ``manifest.json`` de uma etapa do pipeline."""

    def __init__(self, stage: str, output_dir: Path | str):
        self.stage = stage
        self.output_dir = Path(output_dir)
        self.path = self.output_dir / MANIFEST_FILENAME

    # ── Escrita ───────────────────────────────────────────────

    def write(
        self,
        output_hash: str,
        config_hash: str = "",
        parent: Optional[PipelineManifest] = None,
    ) -> Dict[str, Any]:
        """Escreve o manifest no diretório de saída.

        Parameters
        ----------
        output_hash
            Hash dos ficheiros de saída chave desta etapa.
        config_hash
            Hash da configuração usada (opcional).
        parent
            Manifest da etapa anterior (``None`` para a primeira etapa).

        Returns
        -------
        O dicionário escrito em disco.
        """
        data: Dict[str, Any] = {
            "stage": self.stage,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config_hash": config_hash,
            "output_hash": output_hash,
            "parent": parent.read_identity() if parent else None,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Manifest escrito: %s (hash=%s…)", self.path, output_hash[:12])
        return data

    # ── Leitura ───────────────────────────────────────────────

    def read(self) -> Optional[Dict[str, Any]]:
        """Lê o manifest completo, ou ``None`` se não existir."""
        if not self.path.exists():
            return None
        return json.loads(self.path.read_text(encoding="utf-8"))

    def read_identity(self) -> Optional[Dict[str, str]]:
        """Retorna ``{"stage": ..., "output_hash": ...}`` ou ``None``."""
        data = self.read()
        if data is None:
            return None
        return {"stage": data["stage"], "output_hash": data["output_hash"]}

    @property
    def output_hash(self) -> Optional[str]:
        """Hash de saída registado, ou ``None`` se não existir."""
        data = self.read()
        return data["output_hash"] if data else None

    @property
    def exists(self) -> bool:
        return self.path.exists()

    # ── Verificação ───────────────────────────────────────────

    def verify_parent(self, parent: PipelineManifest) -> bool:
        """``True`` se o parent actual coincide com o registado no manifest."""
        data = self.read()
        if data is None:
            return False
        recorded_parent = data.get("parent")
        current_parent = parent.read_identity()
        return recorded_parent == current_parent

    @staticmethod
    def verify_chain(*manifests: PipelineManifest) -> None:
        """Verifica a cadeia completa de manifests.

        Parameters
        ----------
        *manifests
            Manifests ordenados da raiz para a folha
            (ex.: segmentation, profiles, dataset).

        Raises
        ------
        StaleArtifactError
            Se alguma ligação na cadeia estiver inconsistente.
        FileNotFoundError
            Se o manifest de uma etapa não existir.
        """
        for i, manifest in enumerate(manifests):
            if not manifest.exists:
                raise FileNotFoundError(
                    f"Manifest não encontrado para '{manifest.stage}': {manifest.path}"
                )
            if i > 0:
                parent = manifests[i - 1]
                if not manifest.verify_parent(parent):
                    recorded = (manifest.read() or {}).get("parent")
                    current = parent.read_identity()
                    raise StaleArtifactError(
                        f"Etapa '{manifest.stage}' depende de '{parent.stage}' "
                        f"com hash {recorded}, mas o actual é {current}. "
                        f"Re-execute as etapas anteriores primeiro."
                    )

    # ── Utilitários de hash ───────────────────────────────────

    @staticmethod
    def hash_files(*paths: Path) -> str:
        """SHA-256 do conteúdo concatenado dos ficheiros (ordenados por nome)."""
        h = hashlib.sha256()
        for p in sorted(paths):
            if p.exists():
                h.update(p.read_bytes())
        return h.hexdigest()

    @staticmethod
    def hash_config(config: Any) -> str:
        """SHA-256 de uma configuração serializável a JSON."""
        return hashlib.sha256(
            json.dumps(config, sort_keys=True, default=str).encode()
        ).hexdigest()

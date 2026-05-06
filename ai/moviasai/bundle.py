"""
Bundle versionado de artefactos de produção.

Cada execução de treinamento/otimização cria um bundle completo em:
    ``models/forecasting/{target}/{version_id}/``

O bundle contém tudo o necessário para reconstruir o pipeline de
predição (ONNX + classificadores + configs), garantindo consistência
mesmo que as etapas anteriores sejam re-executadas.

Estrutura
---------
::

    {version_id}/
    ├── model.onnx                  # Modelo de forecasting
    ├── classifiers/
    │   ├── stage1_type_BEST.onnx   # Classificador de tipo
    │   ├── stage2_km_BEST.onnx     # Segmentador KM
    │   └── stage2_h_BEST.onnx      # Segmentador H
    ├── config/                     # Configs usadas
    │   ├── segmentation_config.yaml
    │   ├── dataset_config.yaml
    │   ├── ...
    │   └── output_config.yaml
    └── manifest.json               # Metadados de versionamento
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _generate_version_id(model_type: str) -> str:
    """Gera identificador de versão: ``YYYYMMDD_HHMMSS_{model_type}_{4hex}``."""
    import secrets

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = secrets.token_hex(2)  # 4 chars hex → 65536 combinações
    return f"{ts}_{model_type}_{suffix}"


class ModelBundle:
    """Cria e carrega bundles versionados de produção."""

    MODEL_FILENAME = "model.onnx"
    CLASSIFIERS_DIR = "classifiers"
    CONFIG_DIR = "config"
    MANIFEST_FILENAME = "manifest.json"

    def __init__(self, bundle_dir: Path):
        self.bundle_dir = Path(bundle_dir)

    @property
    def model_path(self) -> Path:
        return self.bundle_dir / self.MODEL_FILENAME

    @property
    def classifiers_dir(self) -> Path:
        return self.bundle_dir / self.CLASSIFIERS_DIR

    @property
    def config_dir(self) -> Path:
        return self.bundle_dir / self.CONFIG_DIR

    @property
    def manifest_path(self) -> Path:
        return self.bundle_dir / self.MANIFEST_FILENAME

    @property
    def version_id(self) -> str:
        return self.bundle_dir.name

    @property
    def exists(self) -> bool:
        return self.model_path.exists()

    # ── Criação ───────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        base_dir: Path,
        target: str,
        model_type: str,
        onnx_source: Path,
        segmentation_models_dir: Path,
        config_dir: Path,
        version_id: Optional[str] = None,
    ) -> "ModelBundle":
        """Cria um bundle versionado copiando todos os artefactos.

        Parameters
        ----------
        base_dir
            Directoria base de forecasting (ex.: ``models/forecasting``).
        target
            ``'km'`` ou ``'h'``.
        model_type
            ``'moe'`` ou ``'multihead'``.
        onnx_source
            Path do modelo ONNX exportado pelo training pipeline.
        segmentation_models_dir
            Directoria dos classificadores gerados pela segmentação
            (ex.: ``logs/segmentation/models/``).
        config_dir
            Directoria dos ficheiros YAML de configuração.
        version_id
            Identificador da versão. Se None, gera automaticamente.

        Returns
        -------
        ModelBundle
            Bundle criado.
        """
        if version_id is None:
            version_id = _generate_version_id(model_type)

        bundle_dir = base_dir / target / version_id
        bundle = cls(bundle_dir)

        bundle_dir.mkdir(parents=True, exist_ok=True)

        # 1. Copiar modelo ONNX (+ ficheiro .data externo se existir)
        shutil.copy2(str(onnx_source), str(bundle.model_path))
        onnx_data = Path(str(onnx_source) + ".data")
        if onnx_data.exists():
            shutil.copy2(str(onnx_data), str(bundle.model_path) + ".data")
        logger.info("Bundle: modelo copiado → %s", bundle.model_path)

        # 2. Copiar classificadores
        bundle.classifiers_dir.mkdir(parents=True, exist_ok=True)
        cls._copy_classifiers(segmentation_models_dir, bundle.classifiers_dir)

        # 3. Copiar configs
        bundle.config_dir.mkdir(parents=True, exist_ok=True)
        cls._copy_configs(config_dir, bundle.config_dir)

        # 4. Escrever manifest
        import hashlib

        model_hash = hashlib.sha256(bundle.model_path.read_bytes()).hexdigest()
        manifest = {
            "version_id": version_id,
            "target": target,
            "model_type": model_type,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_hash": model_hash,
        }
        bundle.manifest_path.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        logger.info("Bundle criado: %s", bundle_dir)

        return bundle

    @staticmethod
    def _copy_classifiers(src_dir: Path, dst_dir: Path) -> None:
        """Copia classificadores stage1 e stage2 (.onnx, .pkl e .json auxiliares)."""
        for stage in ("stage1", "stage2"):
            stage_src = src_dir / stage
            if not stage_src.exists():
                continue
            for f in sorted(stage_src.iterdir()):
                if f.suffix in (".onnx", ".pkl", ".json"):
                    shutil.copy2(str(f), str(dst_dir / f.name))
                    logger.debug("  classifier: %s", f.name)

    @staticmethod
    def _copy_configs(src_dir: Path, dst_dir: Path) -> None:
        """Copia todos os YAML de configuração."""
        for f in sorted(src_dir.iterdir()):
            if f.suffix in (".yaml", ".yml"):
                shutil.copy2(str(f), str(dst_dir / f.name))

    # ── Leitura ───────────────────────────────────────────────

    def read_manifest(self) -> Dict:
        """Lê o manifest do bundle."""
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def classifier_path(self, name: str) -> Path:
        """Path de um classificador dentro do bundle (ex.: ``stage2_km_BEST.onnx``)."""
        return self.classifiers_dir / name

    def config_path(self, name: str) -> Path:
        """Path de um ficheiro de configuração dentro do bundle."""
        return self.config_dir / name

    # ── Listagem ──────────────────────────────────────────────

    @classmethod
    def list_versions(cls, base_dir: Path, target: str) -> List["ModelBundle"]:
        """Lista bundles disponíveis para um target, ordenados por data (mais recente primeiro)."""
        target_dir = base_dir / target
        if not target_dir.exists():
            return []
        bundles = []
        for d in sorted(target_dir.iterdir(), reverse=True):
            if d.is_dir() and (d / cls.MODEL_FILENAME).exists():
                bundles.append(cls(d))
        return bundles

    @classmethod
    def get_by_version(cls, base_dir: Path, target: str, version_id: str) -> "ModelBundle":
        """Carrega um bundle por version_id."""
        bundle = cls(base_dir / target / version_id)
        if not bundle.exists:
            raise FileNotFoundError(
                f"Bundle não encontrado: {bundle.bundle_dir}"
            )
        return bundle

"""
Preditor de produção para os modelos de forecasting.

Recebe um :class:`DataInput` já normalizado, monta tensores,
executa inferência ONNX e devolve predições denormalizadas.

Suporta automaticamente ambos os modelos (MultiHead e SAFE MoE) —
detecta o tipo a partir dos input names do ONNX.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import onnxruntime as ort
import pandas as pd

from moviasai.data.data_loaders import ForecastingDataModule, MoEForecastingDataModule
from moviasai.data.dataset import DataInput
from moviasai.data.normalization import ProfileDatasetNormalizer


class ForecastingPredictor:
    """
    Pipeline de inferência: monta tensores → ONNX → denormaliza.

    Detecta automaticamente se o modelo ONNX é MultiHead (2 inputs) ou
    SAFE MoE (3 inputs) a partir dos ``input_names`` do ONNX.

    Parameters
    ----------
    onnx_path : str | Path
        Caminho do modelo ``.onnx``.
    horizon_weeks : int | List[int]
        Horizonte de previsão em semanas (ex.: ``4`` ou ``[1, 1, 2]``).
        Se inteiro, gera lista de 1s (ex.: ``4`` → ``[1,1,1,1]``).
    """

    def __init__(
        self,
        onnx_path: str | Path,
        horizon_weeks: Union[int, List[int]],
    ):
        head_weeks = [1] * horizon_weeks if isinstance(horizon_weeks, int) else list(horizon_weeks)
        self.head_days = np.array([w * 7 for w in head_weeks], dtype=np.float64)

        self._session = ort.InferenceSession(
            str(onnx_path),
            providers=ort.get_available_providers(),
        )

        # Auto-detectar tipo de modelo
        self._input_names = [inp.name for inp in self._session.get_inputs()]
        self._is_moe = "X_cluster_proba" in self._input_names

    @classmethod
    def from_config(
        cls,
        target: str,
        predictor_cfg,
        output_cfg,
        horizon_weeks: Union[int, List[int]],
    ) -> "ForecastingPredictor":
        """
        Constrói o predictor a partir de configs.

        Parameters
        ----------
        target : str
            Métrica alvo: ``'km'`` ou ``'h'``.
        predictor_cfg : PredictorConfig
            Configuração com nomes dos modelos ONNX por target.
        output_cfg : OutputConfig
            Configuração de paths (resolve forecasting dir).
        horizon_weeks : int | List[int]
            Horizonte em semanas (do dataset_config).
        """
        onnx_path = predictor_cfg.model_path(target, output_cfg.models.forecasting)

        return cls(
            onnx_path=onnx_path,
            horizon_weeks=horizon_weeks,
        )

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def predict(self, data_input: DataInput) -> Dict[str, np.ndarray]:
        """
        Executa predição sobre dados **já normalizados**.

        Parameters
        ----------
        data_input : DataInput
            Dados de entrada normalizados.

        Returns
        -------
        dict
            ``y_heads`` : np.ndarray (N, H) — predições agregadas denormalizadas
            ``y_daily`` : np.ndarray (N, D) — predições diárias denormalizadas
            ``y_heads_norm`` : np.ndarray (N, H) — saída crua normalizada
            ``y_daily_norm`` : np.ndarray (N, D) — saída crua normalizada
        """
        upper = np.asarray(data_input.upper, dtype=np.float64)
        X_rec_norm = data_input.X_recent.values.astype(np.float32)

        # 1. Montar tensores
        feeds = self._build_feeds(data_input.X_general, X_rec_norm)

        # 2. Inferência ONNX
        y_heads_norm, y_daily_norm = self._run_onnx(feeds)

        # Garantir não-negatividade (pós-processamento simples)
        y_heads_norm = np.clip(y_heads_norm, a_min=0.0, a_max=None)
        y_daily_norm = np.clip(y_daily_norm, a_min=0.0, a_max=None)

        # 3. Denormalizar
        y_heads = ProfileDatasetNormalizer.inverse_transform_target_heads(
            y_heads_norm, upper, self.head_days.tolist(),
        )
        y_daily = ProfileDatasetNormalizer.inverse_transform_target_daily(
            y_daily_norm, upper,
        )

        return {
            "y_heads": y_heads,
            "y_daily": y_daily,
            "y_heads_norm": y_heads_norm,
            "y_daily_norm": y_daily_norm,
        }

    # ------------------------------------------------------------------
    # Passos internos
    # ------------------------------------------------------------------

    def _build_feeds(
        self,
        X_gen_norm: pd.DataFrame,
        X_rec_norm: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Monta dict de feeds ONNX, adaptando-se ao tipo de modelo."""
        if self._is_moe:
            X_struct, X_cluster, X_recent_3d = MoEForecastingDataModule.build_inference_tensors(
                X_gen_norm, X_rec_norm,
            )
            return {
                "X_general_struct": X_struct,
                "X_cluster_proba": X_cluster,
                "X_recent": X_recent_3d,
            }
        else:
            X_struct, X_recent_3d = ForecastingDataModule.build_inference_tensors(
                X_gen_norm, X_rec_norm,
            )
            return {
                "X_general": X_struct,
                "X_recent": X_recent_3d,
            }

    def _run_onnx(self, feeds: Dict[str, np.ndarray]):
        """Executa sessão ONNX e retorna (y_heads, y_daily)."""
        outputs = self._session.run(None, feeds)
        return outputs[0], outputs[1]

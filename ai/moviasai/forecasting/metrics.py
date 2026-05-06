"""Métricas de decisão de manutenção (horizonte misto)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def find_crossing_day(
    daily_values: np.ndarray,
    head_values: np.ndarray,
    head_days: list,
    limit: float,
) -> float | None:
    """
    Estima em que *dia* (float, 1-indexed) o acumulado cruza ``limit``.

    Resolução:
    - Bloco 1 (head 0): usa ``daily_values`` dia a dia.
    - Blocos seguintes: usa ``head_values`` como bloco semanal,
      assumindo distribuição uniforme **apenas para interpolar
      dentro da semana** em que ocorre o cruzamento.

    Retorna ``None`` se o limite não for atingido no horizonte.
    """
    D = len(daily_values)                      # daily_horizon (ex.: 7)
    first_head_days = head_days[0]             # ex.: 7

    # --- Fase 1: varrer daily_values (semana 1) ---
    cumsum = 0.0
    for d in range(min(D, first_head_days)):
        cumsum += daily_values[d]
        if cumsum >= limit:
            return float(d + 1)                # dia 1-indexed

    # --- Fase 2: varrer heads restantes como blocos ---
    offset_day = first_head_days
    cumsum_at_start = cumsum           # head 0 = semana 1 completa
    for h in range(1, len(head_days)):
        cumsum_at_end = cumsum_at_start + head_values[h]
        if cumsum_at_end >= limit:
            # Interpolar linearmente dentro deste bloco
            remaining = limit - cumsum_at_start
            if head_values[h] > 0:
                frac = remaining / head_values[h]
            else:
                frac = 1.0
            return offset_day + frac * head_days[h]
        cumsum_at_start = cumsum_at_end
        offset_day += head_days[h]

    return None


def compute_maintenance_metrics(
    y_daily_true: np.ndarray,
    y_daily_pred: np.ndarray,
    y_heads_true: np.ndarray,
    y_heads_pred: np.ndarray,
    head_days: list,
    upper: np.ndarray,
) -> Dict:
    """Calcula métricas de decisão de manutenção para k=2,3,4 semanas.

    Usa resolução **mista**: ``y_daily`` para a 1ª semana (granularidade
    diária) e ``y_heads`` para as semanas seguintes (agregado semanal).
    Nunca fabrica resolução diária além de ``daily_horizon``.

    Parameters
    ----------
    y_daily_true, y_daily_pred : (N, D) consumo diário denormalizado
    y_heads_true, y_heads_pred : (N, H) consumo agregado denormalizado
    head_days : dias por head (ex.: [7, 7, 7, 7])
    upper : (N,) P95 diário por veículo (escala original)

    Returns
    -------
    Dict com chaves ``'k2'``, ``'k3'``, ``'k4'``.
    """
    total_days = sum(head_days)
    N = len(upper)
    n_heads = len(head_days)
    results: Dict = {}

    for k in range(2, n_heads + 1):
        limits = k * upper * head_days[k - 1]
        errors = np.full(N, np.nan)

        for i in range(N):
            real_day = find_crossing_day(
                y_daily_true[i], y_heads_true[i], head_days, limits[i],
            )
            pred_day = find_crossing_day(
                y_daily_pred[i], y_heads_pred[i], head_days, limits[i],
            )

            if real_day is not None and pred_day is not None:
                errors[i] = pred_day - real_day
            elif real_day is None and pred_day is None:
                errors[i] = 0
            elif pred_day is not None:
                errors[i] = pred_day - (total_days + 1)
            else:
                errors[i] = (total_days + 1) - real_day

        results[f"k{k}"] = {
            "mean_error": float(np.nanmean(errors)),
            "mae_days": float(np.nanmean(np.abs(errors))),
            "p90_error": float(np.nanpercentile(errors, 90)),
            "pct_late": float(np.nanmean(errors > 0) * 100),
            "pct_early": float(np.nanmean(errors < 0) * 100),
            "errors": errors,
        }

    return results


def save_predictions(
    pred_dir: str | Path,
    split_name: str,
    metadata: pd.DataFrame,
    y_heads_true: np.ndarray,
    y_heads_pred: np.ndarray,
    y_daily_true: np.ndarray,
    y_daily_pred: np.ndarray,
) -> Path:
    """Salva predições como arquivos numpy + metadados CSV.

    Gera dentro de ``pred_dir``:
    - ``metadata_{split}.csv`` (veiculo_id, cluster, upper)
    - ``predicted_heads_{split}.npy``
    - ``real_heads_{split}.npy``
    - ``predicted_days_{split}.npy``
    - ``real_days_{split}.npy``

    Parameters
    ----------
    pred_dir : str ou Path
        Diretório de saída (será criado se não existir).
    split_name : str
        Nome do split (``'val'`` ou ``'test'``).
    metadata : pd.DataFrame
        Metadados das amostras (deve conter ``veiculo_id``, ``cluster``, ``upper``).
    y_heads_true, y_heads_pred : (N, H) ndarray
        Valores reais/previstos denormalizados por head.
    y_daily_true, y_daily_pred : (N, D) ndarray
        Valores reais/previstos denormalizados diários.

    Returns
    -------
    Path do diretório.
    """
    d = Path(pred_dir)
    d.mkdir(parents=True, exist_ok=True)

    meta_cols = ["veiculo_id", "upper"]
    if "cluster" in metadata.columns:
        meta_cols.insert(1, "cluster")
    meta_df = metadata[meta_cols].reset_index(drop=True)
    meta_df.to_csv(d / f"metadata_{split_name}.csv", index=False)

    np.save(d / f"predicted_heads_{split_name}.npy", y_heads_pred)
    np.save(d / f"real_heads_{split_name}.npy", y_heads_true)
    np.save(d / f"predicted_days_{split_name}.npy", y_daily_pred)
    np.save(d / f"real_days_{split_name}.npy", y_daily_true)

    return d


def load_predictions(pred_dir: str | Path, split: str) -> Dict:
    """Carrega predições salvas por ``TrainingPipeline._save_predictions``.

    Parameters
    ----------
    pred_dir : str ou Path
        Diretório ``predictions/`` (ex.: ``logs/training/km/predictions``).
    split : str
        Nome do split (``'val'`` ou ``'test'``).

    Returns
    -------
    Dict com chaves:
        - ``metadata`` : pd.DataFrame (veiculo_id, cluster, upper)
        - ``predicted_heads`` : (N, H) ndarray
        - ``real_heads`` : (N, H) ndarray
        - ``predicted_days`` : (N, D) ndarray
        - ``real_days`` : (N, D) ndarray
    """
    d = Path(pred_dir)
    return {
        "metadata": pd.read_csv(d / f"metadata_{split}.csv"),
        "predicted_heads": np.load(d / f"predicted_heads_{split}.npy"),
        "real_heads": np.load(d / f"real_heads_{split}.npy"),
        "predicted_days": np.load(d / f"predicted_days_{split}.npy"),
        "real_days": np.load(d / f"real_days_{split}.npy"),
    }

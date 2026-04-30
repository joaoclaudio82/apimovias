from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import polars as pl

_MANDATORY_COLS = ["veiculo_id", "data"]
_VALID_TARGETS = ("h", "km")


def load_raw_data(
    path: Union[str, Path],
    target: Optional[str] = None,
) -> pl.DataFrame:
    """Lê o ficheiro de dados brutos e devolve um DataFrame Polars.

    Parameters
    ----------
    path : str | Path
        Caminho para o ficheiro CSV.
    target : str | None
        ``'h'``, ``'km'`` ou ``None``.  Se ``None``, carrega ambas
        as colunas de target.

    Returns
    -------
    pl.DataFrame
        DataFrame com as colunas obrigatórias e as colunas de target.
    """
    if target is not None and target not in _VALID_TARGETS:
        raise ValueError(f"target deve ser um de {_VALID_TARGETS}, recebeu {target!r}")

    if target is None:
        target_cols = [f"{t}_dia_clean" for t in _VALID_TARGETS]
    else:
        target_cols = [f"{target}_dia_clean"]

    columns = _MANDATORY_COLS + target_cols

    df = pl.read_csv(path, columns=columns, try_parse_dates=True)
    return df


def expand_cluster_columns(
    predictions: np.ndarray,
    probabilities: Optional[np.ndarray],
    cluster_features: Optional[str],
    target: str,
) -> pd.DataFrame:
    """Gera DataFrame com colunas cluster_0_{target}..cluster_n_{target}.

    Parameters
    ----------
    predictions : np.ndarray, shape (k,)
        Predição de cluster por veículo.
    probabilities : np.ndarray ou None, shape (k, n)
        Probabilidades por classe. Necessário quando
        *cluster_features* == ``'probabilities'``.
    cluster_features : str ou None
        ``'prediction'`` — one-hot a partir de *predictions*.
        ``'probabilities'`` — colunas com probabilidades.
        ``None`` — devolve DataFrame vazio.
    target : str
        ``'km'`` ou ``'h'``.

    Returns
    -------
    pd.DataFrame
        DataFrame com as colunas de cluster (sem índice nomeado).
    """
    if cluster_features == 'prediction':
        unique_clusters = sorted(set(predictions[~np.isnan(predictions)].astype(int)))
        data = {
            f'cluster_{c}_{target}': (predictions == c).astype(float)
            for c in unique_clusters
        }
        return pd.DataFrame(data)
    elif cluster_features == 'probabilities':
        if probabilities is None:
            return pd.DataFrame()
        data = {
            f'cluster_{i}_{target}': probabilities[:, i]
            for i in range(probabilities.shape[1])
        }
        return pd.DataFrame(data)
    return pd.DataFrame()


def train_test_split(
    dataset,
    test_size=None,
    train_size=None,
    random_state=None,
    shuffle=True,
) -> tuple:
    """
    Divide um ProfileDataset em treino e teste por corte temporal em ref_date.

    O split garante que todas as ref_dates de treino são anteriores às de teste.
    Como a divisão é por semana, as proporções finais são aproximadas.

    Parameters
    ----------
    dataset : ProfileDataset
        Dataset a dividir
    test_size : float ou int, optional
        Proporção (0–1) ou número absoluto de amostras para teste.
        Default: 0.25 se train_size também for None.
    train_size : float ou int, optional
        Proporção (0–1) ou número absoluto de amostras para treino.
    random_state : int, optional
        Seed para shuffle
    shuffle : bool
        Se True, embaralha amostras dentro de cada subset

    Returns
    -------
    (ProfileDataset, ProfileDataset)
        (train_dataset, test_dataset)
    """
    from moviasai.data.dataset import ProfileDataset

    n = len(dataset)
    ref_dates = pd.to_datetime(dataset.metadata['ref_date'])
    unique_dates = np.sort(ref_dates.unique())

    # Contagem cumulativa por ref_date
    counts = np.array([int((ref_dates == d).sum()) for d in unique_dates])
    cumsum = np.cumsum(counts)

    # Determinar número alvo de amostras de teste
    if test_size is None and train_size is None:
        target_test = int(round(n * 0.25))
    elif test_size is not None:
        target_test = int(round(n * test_size)) if isinstance(test_size, float) else test_size
    else:
        target_train = int(round(n * train_size)) if isinstance(train_size, float) else train_size
        target_test = n - target_train

    target_test = max(1, min(target_test, n - 1))

    # Encontrar melhor ponto de corte (treino = datas até cutoff_idx inclusive)
    best_idx = 0
    best_diff = float('inf')
    for i in range(len(unique_dates)):
        n_test_i = n - cumsum[i]
        diff = abs(n_test_i - target_test)
        if diff < best_diff:
            best_diff = diff
            best_idx = i

    cutoff_date = unique_dates[best_idx]
    train_mask = (ref_dates <= cutoff_date).values
    test_mask = ~train_mask

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]

    if shuffle:
        rng = np.random.RandomState(random_state)
        rng.shuffle(train_idx)
        rng.shuffle(test_idx)

    def _subset(idx):
        return ProfileDataset(
            metadata=dataset.metadata.iloc[idx].reset_index(drop=True),
            X_general=dataset.X_general.iloc[idx].reset_index(drop=True),
            X_recent=dataset.X_recent.iloc[idx].reset_index(drop=True),
            y_heads=dataset.y_heads.iloc[idx].reset_index(drop=True),
            y_daily=dataset.y_daily.iloc[idx].reset_index(drop=True),
            weights=dataset.weights[idx],
            config=dataset.config,
        )

    ds_train = _subset(train_idx)
    ds_test = _subset(test_idx)

    # Log
    train_dates = pd.to_datetime(ds_train.metadata['ref_date'])
    test_dates = pd.to_datetime(ds_test.metadata['ref_date'])
    print(f"train_test_split temporal:")
    print(f"  Cutoff:       {pd.Timestamp(cutoff_date).date()}")
    print(f"  Treino:       {len(ds_train):,} amostras ({len(ds_train)/n*100:.1f}%) "
          f"| {pd.Timestamp(train_dates.min()).date()} → {pd.Timestamp(train_dates.max()).date()} "
          f"({train_dates.nunique()} semanas)")
    print(f"  Teste:        {len(ds_test):,} amostras ({len(ds_test)/n*100:.1f}%) "
          f"| {pd.Timestamp(test_dates.min()).date()} → {pd.Timestamp(test_dates.max()).date()} "
          f"({test_dates.nunique()} semanas)")

    return ds_train, ds_test

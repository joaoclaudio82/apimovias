import re
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from moviasai.data.dataset import ProfileDataset


class ForecastingDataset(Dataset):
    """Dataset para previsão de consumo de veículos."""

    def __init__(
        self,
        X_general: torch.Tensor,
        X_recent: torch.Tensor,
        y_heads: torch.Tensor,
        y_daily: torch.Tensor,
        weights: torch.Tensor,
    ):
        self.X_general = X_general
        self.X_recent = X_recent
        self.y_heads = y_heads
        self.y_daily = y_daily
        self.weights = weights

    def __len__(self):
        return len(self.X_general)

    def __getitem__(self, idx):
        return (
            self.X_general[idx],
            self.X_recent[idx],
            self.y_heads[idx],
            self.y_daily[idx],
            self.weights[idx],
        )


class ForecastingDataModule(pl.LightningDataModule):
    """
    DataModule para o modelo de previsão.

    - Remove features ``day_*`` de ``X_general`` (perfil estrutural)
    - Constrói ``X_recent`` 3D ``(N, T, F)`` combinando valores diários
      com features ``day_*`` alinhadas seg→dom

    Parameters
    ----------
    train_dataset : ProfileDataset
        Dataset normalizado de treino
    val_dataset : ProfileDataset, optional
        Dataset normalizado de validação
    test_dataset : ProfileDataset, optional
        Dataset normalizado de teste
    batch_size : int
        Tamanho do batch
    num_workers : int
        Workers para DataLoader
    """

    def __init__(
        self,
        train_dataset: ProfileDataset,
        val_dataset: Optional[ProfileDataset] = None,
        test_dataset: Optional[ProfileDataset] = None,
        batch_size: int = 64,
        num_workers: int = 0,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Separar features estruturais de day_* (usando colunas do treino)
        self.struct_cols, self.day_features = self._parse_columns(
            train_dataset.X_general.columns
        )

        self._train_ds = self._to_forecasting_dataset(train_dataset)
        self._val_ds = self._to_forecasting_dataset(val_dataset) if val_dataset is not None else None
        self._test_ds = self._to_forecasting_dataset(test_dataset) if test_dataset is not None else None

        # Dimensões expostas para configuração do modelo
        self.n_general_features = self._train_ds.X_general.shape[1]
        self.n_recent_steps = self._train_ds.X_recent.shape[1]
        self.n_recent_features = self._train_ds.X_recent.shape[2]
        self.n_heads = self._train_ds.y_heads.shape[1]
        self.n_daily = self._train_ds.y_daily.shape[1]

    def _to_forecasting_dataset(self, dataset: ProfileDataset) -> ForecastingDataset:
        """Converte ProfileDataset em ForecastingDataset (tensores torch)."""
        X_struct = dataset.X_general[self.struct_cols].values.astype(np.float32)
        X_recent_3d = self._build_X_recent_3d(
            dataset.X_recent, dataset.X_general, self.day_features
        )
        return ForecastingDataset(
            X_general=torch.from_numpy(X_struct),
            X_recent=torch.from_numpy(X_recent_3d),
            y_heads=torch.from_numpy(np.asarray(dataset.y_heads, dtype=np.float32)),
            y_daily=torch.from_numpy(np.asarray(dataset.y_daily, dtype=np.float32)),
            weights=torch.from_numpy(np.asarray(dataset.weights, dtype=np.float32)),
        )

    # ------------------------------------------------------------------
    # Preparação de colunas
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_columns(columns):
        """
        Separa colunas estruturais das ``day_*``.

        Returns
        -------
        struct_cols : list[str]
            Colunas que NÃO são ``day_*``
        day_features : dict[int, list[str]]
            ``{dia_semana: [col_name, ...]}`` com ordem consistente
        """
        day_pattern = re.compile(r"^day_(\d+)_(.+)$")
        struct_cols = []
        day_groups: dict[int, dict[str, str]] = defaultdict(dict)

        for col in columns:
            m = day_pattern.match(col)
            if m:
                day_num = int(m.group(1))
                feat_suffix = m.group(2)
                day_groups[day_num][feat_suffix] = col
            else:
                struct_cols.append(col)

        if day_groups:
            feat_order = sorted(day_groups[1].keys())
            day_features = {
                d: [day_groups[d][f] for f in feat_order] for d in range(1, 8)
            }
        else:
            day_features = {}

        return struct_cols, day_features

    # ------------------------------------------------------------------
    # Construção do X_recent 3D
    # ------------------------------------------------------------------

    @staticmethod
    def _build_X_recent_3d(
        X_recent_norm: np.ndarray,
        X_general_norm: pd.DataFrame,
        day_features: dict,
    ) -> np.ndarray:
        """
        Constrói tensor 3D ``(N, T, F)`` para o encoder temporal.

        Cada timestep ``t`` contém:
        - Valor diário normalizado (1 feature)
        - Features ``day_*`` do dia da semana correspondente (K features)

        Alinhamento: ``t=0`` → segunda (day_1) … ``t=6`` → domingo (day_7),
        repetindo semanalmente.
        """
        if isinstance(X_recent_norm, pd.DataFrame):
            X_recent_norm = X_recent_norm.values

        N, T = X_recent_norm.shape

        if not day_features:
            return X_recent_norm.reshape(N, T, 1).astype(np.float32)

        K = len(day_features[1])
        F = 1 + K
        X_3d = np.zeros((N, T, F), dtype=np.float32)

        # Feature 0: valor diário
        X_3d[:, :, 0] = X_recent_norm.astype(np.float32)

        # Pré-extrair arrays por dia da semana: (N, K)
        day_arrays = {
            d: X_general_norm[cols].values.astype(np.float32)
            for d, cols in day_features.items()
        }

        # Preencher features day_* por timestep
        for t in range(T):
            weekday = (t % 7) + 1  # 1=segunda … 7=domingo
            X_3d[:, t, 1:] = day_arrays[weekday]

        return X_3d

    # ------------------------------------------------------------------
    # Construção de tensores para inferência
    # ------------------------------------------------------------------

    @staticmethod
    def build_inference_tensors(
        X_general_norm: pd.DataFrame,
        X_recent_norm: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Monta tensores de inferência a partir de dados já normalizados.

        Separa features estruturais de ``day_*`` em ``X_general_norm``
        e constrói ``X_recent_3d``.

        Parameters
        ----------
        X_general_norm : pd.DataFrame (N, F)
            Features normalizadas (pode conter ``day_*``).
        X_recent_norm : np.ndarray (N, T)
            Histórico recente normalizado.

        Returns
        -------
        X_struct : np.ndarray (N, F_struct)
            Features estruturais (sem ``day_*``), float32.
        X_recent_3d : np.ndarray (N, T, 1+K)
            Tensor 3D com valor diário + features ``day_*``, float32.
        """
        struct_cols, day_features = ForecastingDataModule._parse_columns(
            X_general_norm.columns,
        )
        X_struct = X_general_norm[struct_cols].values.astype(np.float32)
        X_recent_3d = ForecastingDataModule._build_X_recent_3d(
            X_recent_norm, X_general_norm, day_features,
        )
        return X_struct, X_recent_3d

    # ------------------------------------------------------------------
    # Setup e DataLoaders
    # ------------------------------------------------------------------

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = self._train_ds
        self.val_dataset = self._val_ds
        self.test_dataset = self._test_ds

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            raise RuntimeError("Nenhum val_dataset foi fornecido.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            raise RuntimeError("Nenhum test_dataset foi fornecido.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return self.test_dataloader() if self.test_dataset is not None else self.val_dataloader()

    def summary(self):
        """Imprime resumo dos datasets."""
        total = len(self._train_ds) + (len(self._val_ds) if self._val_ds else 0) + (len(self._test_ds) if self._test_ds else 0)
        print(f"ForecastingDataModule:")
        print(f"  Treino:       {len(self._train_ds):,} amostras ({len(self._train_ds)/total*100:.1f}%)")
        if self._val_ds is not None:
            print(f"  Validação:    {len(self._val_ds):,} amostras ({len(self._val_ds)/total*100:.1f}%)")
        if self._test_ds is not None:
            print(f"  Teste:        {len(self._test_ds):,} amostras ({len(self._test_ds)/total*100:.1f}%)")
        print(f"Dimensões:")
        print(f"  X_general:    {self.n_general_features} features")
        print(f"  X_recent:     ({self.n_recent_steps}, {self.n_recent_features})")
        print(f"  y_heads:      {self.n_heads} cabeças")
        print(f"  y_daily:      {self.n_daily} dias")


# ======================================================================
# SAFE MoE — DataModule e Dataset com 6 tensores
# ======================================================================


class MoEForecastingDataset(Dataset):
    """Dataset com 6 tensores para o SAFE MoE."""

    def __init__(
        self,
        X_general_struct: torch.Tensor,
        X_cluster_proba: torch.Tensor,
        X_recent: torch.Tensor,
        y_heads: torch.Tensor,
        y_daily: torch.Tensor,
        weights: torch.Tensor,
    ):
        self.X_general_struct = X_general_struct
        self.X_cluster_proba = X_cluster_proba
        self.X_recent = X_recent
        self.y_heads = y_heads
        self.y_daily = y_daily
        self.weights = weights

    def __len__(self):
        return len(self.X_general_struct)

    def __getitem__(self, idx):
        return (
            self.X_general_struct[idx],
            self.X_cluster_proba[idx],
            self.X_recent[idx],
            self.y_heads[idx],
            self.y_daily[idx],
            self.weights[idx],
        )


class MoEForecastingDataModule(pl.LightningDataModule):
    """
    DataModule para o SAFE MoE.

    Comportamento idêntico ao ``ForecastingDataModule``, excepto:
    - Separa ``cluster_*`` de ``X_general``
    - Entrega batches de 6 tensores em vez de 5

    Parameters
    ----------
    train_dataset, val_dataset, test_dataset : ProfileDataset
        Datasets normalizados.
    batch_size, num_workers :
        Configuração do DataLoader.
    """

    def __init__(
        self,
        train_dataset: ProfileDataset,
        val_dataset: Optional[ProfileDataset] = None,
        test_dataset: Optional[ProfileDataset] = None,
        batch_size: int = 64,
        num_workers: int = 0,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Reutilizar lógica de parsing do ForecastingDataModule
        self.struct_cols, self.day_features = ForecastingDataModule._parse_columns(
            train_dataset.X_general.columns
        )

        # Identificar colunas de cluster (cluster_0_km, cluster_1_h, ...)
        cluster_re = re.compile(r'^cluster_\d+_(?:km|h)$')
        self.cluster_cols = sorted([
            c for c in self.struct_cols if cluster_re.match(c)
        ])

        # Colunas estruturais puras (sem day_*, sem cluster_N)
        _cluster_set = set(self.cluster_cols)
        self.general_struct_cols = [
            c for c in self.struct_cols if c not in _cluster_set
        ]

        self.n_clusters = len(self.cluster_cols)
        if self.n_clusters == 0:
            raise ValueError(
                "Nenhuma coluna 'cluster_*' encontrada em X_general. "
                "O SAFE MoE requer dataset gerado com cluster_features='prediction' ou 'probabilities'."
            )

        # Converter datasets
        self._train_ds = self._to_moe_dataset(train_dataset)
        self._val_ds = self._to_moe_dataset(val_dataset) if val_dataset is not None else None
        self._test_ds = self._to_moe_dataset(test_dataset) if test_dataset is not None else None

        # Dimensões expostas para configuração do modelo
        self.n_general_features = self._train_ds.X_general_struct.shape[1]
        self.n_recent_steps = self._train_ds.X_recent.shape[1]
        self.n_recent_features = self._train_ds.X_recent.shape[2]
        self.n_heads = self._train_ds.y_heads.shape[1]
        self.n_daily = self._train_ds.y_daily.shape[1]

    def _to_moe_dataset(self, dataset: ProfileDataset) -> MoEForecastingDataset:
        """Converte ProfileDataset em MoEForecastingDataset."""
        X_struct = dataset.X_general[self.general_struct_cols].values.astype(np.float32)
        X_cluster = dataset.X_general[self.cluster_cols].values.astype(np.float32)
        X_recent_3d = ForecastingDataModule._build_X_recent_3d(
            dataset.X_recent, dataset.X_general, self.day_features,
        )
        return MoEForecastingDataset(
            X_general_struct=torch.from_numpy(X_struct),
            X_cluster_proba=torch.from_numpy(X_cluster),
            X_recent=torch.from_numpy(X_recent_3d),
            y_heads=torch.from_numpy(np.asarray(dataset.y_heads, dtype=np.float32)),
            y_daily=torch.from_numpy(np.asarray(dataset.y_daily, dtype=np.float32)),
            weights=torch.from_numpy(np.asarray(dataset.weights, dtype=np.float32)),
        )

    # ------------------------------------------------------------------
    # Construção de tensores para inferência
    # ------------------------------------------------------------------

    @staticmethod
    def build_inference_tensors(
        X_general_norm: pd.DataFrame,
        X_recent_norm: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Monta tensores de inferência para o SAFE MoE (3 arrays).

        Separa ``cluster_*`` de features estruturais e constrói ``X_recent_3d``.

        Returns
        -------
        X_struct : np.ndarray (N, F_struct)
        X_cluster : np.ndarray (N, K)
        X_recent_3d : np.ndarray (N, T, 1+K_day)
        """
        struct_cols, day_features = ForecastingDataModule._parse_columns(
            X_general_norm.columns,
        )
        cluster_re = re.compile(r'^cluster_\d+_(?:km|h)$')
        cluster_cols = sorted(c for c in struct_cols if cluster_re.match(c))
        cluster_set = set(cluster_cols)
        general_struct_cols = [c for c in struct_cols if c not in cluster_set]

        X_struct = X_general_norm[general_struct_cols].values.astype(np.float32)
        X_cluster = X_general_norm[cluster_cols].values.astype(np.float32)
        X_recent_3d = ForecastingDataModule._build_X_recent_3d(
            X_recent_norm, X_general_norm, day_features,
        )
        return X_struct, X_cluster, X_recent_3d

    # ------------------------------------------------------------------
    # Setup e DataLoaders
    # ------------------------------------------------------------------

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = self._train_ds
        self.val_dataset = self._val_ds
        self.test_dataset = self._test_ds

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            raise RuntimeError("Nenhum val_dataset foi fornecido.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            raise RuntimeError("Nenhum test_dataset foi fornecido.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return self.test_dataloader() if self.test_dataset is not None else self.val_dataloader()

    def summary(self):
        total = len(self._train_ds) + (len(self._val_ds) if self._val_ds else 0) + (len(self._test_ds) if self._test_ds else 0)
        print(f"MoEForecastingDataModule:")
        print(f"  Treino:       {len(self._train_ds):,} amostras ({len(self._train_ds)/total*100:.1f}%)")
        if self._val_ds is not None:
            print(f"  Validação:    {len(self._val_ds):,} amostras ({len(self._val_ds)/total*100:.1f}%)")
        if self._test_ds is not None:
            print(f"  Teste:        {len(self._test_ds):,} amostras ({len(self._test_ds)/total*100:.1f}%)")
        print(f"Dimensões:")
        print(f"  X_general_struct:  {self.n_general_features} features")
        print(f"  X_cluster_proba:   {self.n_clusters} clusters")
        print(f"  X_recent:          ({self.n_recent_steps}, {self.n_recent_features})")
        print(f"  y_heads:           {self.n_heads} cabeças")
        print(f"  y_daily:           {self.n_daily} dias")

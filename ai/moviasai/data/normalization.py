from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from moviasai.data.dataset import DataInput, ProfileDataset
from moviasai.profiling.feature_extraction import get_norm_rules


class ProfileDatasetNormalizer:
    """
    Normalizador semântico de features e targets.

    Aplica normalização diferenciada por tipo de feature,
    consultando as regras definidas em cada FeatureExtractor.

    Targets:
    - y_daily / X_recent: clip(y, 0, upper) / upper
    - y_heads: clip(y, 0, upper * dias_bloco) / (upper * dias_bloco)
    """

    def __init__(
        self,
        global_scaler=None,
        cv_max: float = 5.0,
        ratio_max: float = 5.0,
    ):
        """
        Parameters
        ----------
        global_scaler : sklearn-compatible scaler (já ajustado), optional
            Scaler para features do tipo 'global' (ex: StandardScaler().fit(X)).
            Obrigatório se existirem features 'global'.
        cv_max : float
            Limite para clip de CV antes de log1p
        ratio_max : float
            Limite para clip de razões antes de log1p
        """
        self.global_scaler = global_scaler
        self.cv_max = cv_max
        self.ratio_max = ratio_max

        self._feature_types: Dict[str, str] = {}
        self._global_columns: List[str] = []

    def _fit(self, features: Dict[str, List[Tuple[str, List[str]]]]):
        """Ajusta os estados internos a partir do mapeamento de features."""
        self._feature_types = {}
        self._global_columns = []

        for prefix, entries in features.items():
            rules = get_norm_rules(prefix)

            for canonical, expanded_cols in entries:
                norm_type = rules.get(canonical)
                if norm_type is None:
                    raise ValueError(
                        f"Feature '{canonical}' do extractor '{prefix}' "
                        f"não tem norm_rule definida."
                    )
                for col in expanded_cols:
                    self._feature_types[col] = norm_type
                    if norm_type == 'global':
                        self._global_columns.append(col)

        if self._global_columns and self.global_scaler is None:
            raise ValueError(
                f"Existem {len(self._global_columns)} features 'global' "
                f"({self._global_columns[:5]}...) mas nenhum global_scaler foi fornecido."
            )

    def fit_normalize_dataset(self, dataset: ProfileDataset) -> ProfileDataset:
        """
        Ajusta estados internos com dataset.features e normaliza o dataset completo.

        Parameters
        ----------
        dataset : ProfileDataset
            Dataset gerado pelo ProfileDatasetGenerator

        Returns
        -------
        ProfileDataset
            Novo dataset com dados normalizados e config atualizado
        """
        self._fit(dataset.features)

        upper = dataset.metadata['upper'].values.astype(np.float64)
        head_days = np.array(dataset.config['head_days'], dtype=np.float64)

        X_general = dataset.X_general
        feature_columns = [c for c in X_general.columns if c in self._feature_types]

        X_general_norm = self._transform_X_general(X_general, upper, feature_columns)
        X_recent_norm = self._scale_by_upper(self._to_array(dataset.X_recent), upper)
        y_heads_norm = self._transform_y_heads(self._to_array(dataset.y_heads), upper, head_days)
        y_daily_norm = self._scale_by_upper(self._to_array(dataset.y_daily), upper)

        config = {**dataset.config, 'normalized': True}

        return ProfileDataset(
            metadata=dataset.metadata,
            X_general=X_general_norm,
            X_recent=pd.DataFrame(X_recent_norm, columns=dataset.X_recent.columns),
            y_heads=pd.DataFrame(y_heads_norm, columns=dataset.y_heads.columns),
            y_daily=pd.DataFrame(y_daily_norm, columns=dataset.y_daily.columns),
            weights=dataset.weights,
            config=config,
        )

    def fit_normalize_data_input(self, data_input: DataInput) -> DataInput:
        """
        Ajusta estados internos e normaliza um :class:`DataInput`.

        Parameters
        ----------
        data_input : DataInput
            Dados de entrada crus gerados por :class:`GenerateDataInput`.

        Returns
        -------
        DataInput
            Novo DataInput com X_general e X_recent normalizados.
        """
        self._fit(data_input.features)

        upper = np.asarray(data_input.upper, dtype=np.float64)
        feature_columns = [c for c in data_input.X_general.columns if c in self._feature_types]

        X_general_norm = self._transform_X_general(data_input.X_general, upper, feature_columns)
        X_recent_norm = self._scale_by_upper(self._to_array(data_input.X_recent), upper)

        return DataInput(
            X_general=X_general_norm,
            X_recent=pd.DataFrame(X_recent_norm, columns=data_input.X_recent.columns),
            features=data_input.features,
            upper=data_input.upper,
        )

    @staticmethod
    def inverse_transform_target_heads(
        y_heads: np.ndarray,
        upper: np.ndarray,
        head_days: List[int],
    ) -> np.ndarray:
        """
        Denormaliza y_heads.

        Parameters
        ----------
        y_heads : np.ndarray
            Targets agregados normalizados
        upper : np.ndarray
            Valor upper por amostra
        head_days : List[int]
            Dias por cabeça (ex: [7, 7, 14])

        Returns
        -------
        np.ndarray
            y_heads denormalizados
        """
        upper = np.asarray(upper, dtype=np.float64)
        head_days = np.array(head_days, dtype=np.float64)
        return ProfileDatasetNormalizer._inverse_y_heads(
            ProfileDatasetNormalizer._to_array(y_heads), upper, head_days
        )

    @staticmethod
    def inverse_transform_target_daily(
        y_daily: np.ndarray,
        upper: np.ndarray,
    ) -> np.ndarray:
        """
        Denormaliza y_daily.

        Parameters
        ----------
        y_daily : np.ndarray
            Targets diários normalizados
        upper : np.ndarray
            Valor upper por amostra

        Returns
        -------
        np.ndarray
            y_daily denormalizados
        """
        upper = np.asarray(upper, dtype=np.float64)
        return ProfileDatasetNormalizer._unscale_by_upper(
            ProfileDatasetNormalizer._to_array(y_daily), upper
        )

    # ================================================================
    # Internos
    # ================================================================

    @staticmethod
    def _to_array(data):
        if isinstance(data, pd.DataFrame):
            return data.values.astype(np.float64)
        return np.asarray(data, dtype=np.float64)

    def _apply_forward(self, vals, norm_type, upper):
        if norm_type == 'by_upper':
            safe = np.maximum(upper, 1e-8)
            return np.clip(vals, 0, upper) / safe
        if norm_type == 'no_norm':
            return vals.copy()
        if norm_type == 'log1p':
            return np.log1p(np.clip(vals, 0, self.cv_max))
        if norm_type == 'ratio':
            vals = np.nan_to_num(vals, nan=0.0, posinf=self.ratio_max, neginf=0.0)
            return np.log1p(np.clip(vals, 0, self.ratio_max))
        if norm_type == 'clamp':
            return np.clip(vals, -1.0, 1.0)
        if norm_type == 'global':
            return vals.copy()  # handled in _transform_X_general
        return vals.copy()

    # ---- X_general ----

    def _transform_X_general(self, df: pd.DataFrame, upper: np.ndarray, feature_columns: List[str]) -> pd.DataFrame:
        result = pd.DataFrame(index=df.index)

        for col in feature_columns:
            vals = df[col].values.astype(np.float64)
            result[col] = self._apply_forward(vals, self._feature_types[col], upper)

        global_cols = [c for c in feature_columns if self._feature_types[c] == 'global']
        if global_cols:
            result[global_cols] = self.global_scaler.transform(result[global_cols].values)

        # Pass-through: colunas presentes no DataFrame mas sem norm_rule (ex.: cluster)
        passthrough_cols = [c for c in df.columns if c not in self._feature_types]
        for col in passthrough_cols:
            result[col] = df[col].values

        return result[df.columns]

    # ---- X_recent / y_daily ----

    @staticmethod
    def _scale_by_upper(data, upper):
        u = np.maximum(upper.reshape(-1, 1), 1e-8)
        return np.clip(data, 0, u) / u

    @staticmethod
    def _unscale_by_upper(data, upper):
        return data * upper.reshape(-1, 1)

    # ---- y_heads ----

    @staticmethod
    def _transform_y_heads(data, upper, head_days):
        scale = np.maximum(upper.reshape(-1, 1), 1e-8) * head_days.reshape(1, -1)
        return np.clip(data, 0, scale) / scale

    @staticmethod
    def _inverse_y_heads(data, upper, head_days):
        scale = upper.reshape(-1, 1) * head_days.reshape(1, -1)
        return data * scale
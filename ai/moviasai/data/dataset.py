import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from moviasai.data.utils import expand_cluster_columns
from moviasai.profiling.feature_extraction import get_extractor_by_prefix

logger = logging.getLogger(__name__)


@dataclass
class ProfileDataset:
    """
    Container para dataset de perfil gerado pelo ProfileDatasetLoader.
    
    Attributes
    ----------
    metadata : pd.DataFrame
        Metadata por amostra: ['veiculo_id', 'ref_date', 'year', 'week', 'upper']
    X_general : pd.DataFrame
        Features gerais (perfis)
    X_recent : pd.DataFrame
        Histórico recente diário (n_samples, 7 * num_weeks_recent)
    y_heads : pd.DataFrame
        Targets agregados por cabeça
    y_daily : pd.DataFrame
        Targets diários
    weights : np.ndarray
        Pesos das amostras
    features : Dict[str, List[Tuple[str, List[str]]]]
        Mapeamento prefixo → [(canônico, [expandido, ...]), ...]
        Ex: {'seg': [('cv_gaps', ['seg_cv_gaps_km'])], 'day': [('mean', ['day_1_mean_km', ...])]}
    config : dict
        Configurações utilizadas para gerar o dataset
    """
    metadata: pd.DataFrame
    X_general: pd.DataFrame
    X_recent: pd.DataFrame
    y_heads: pd.DataFrame
    y_daily: pd.DataFrame
    weights: np.ndarray
    features: Dict[str, List[Tuple[str, List[str]]]] = field(default_factory=dict)
    config: Dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.X_general)

    def __iter__(self):
        """Permite unpacking: metadata, X_general, X_recent, y_heads, y_daily, weights = dataset"""
        return iter((self.metadata, self.X_general, self.X_recent, self.y_heads, self.y_daily, self.weights))

    def __repr__(self) -> str:
        metric = self.config.get('metric', '?')
        return (
            f"ProfileDataset(metric='{metric}', n_samples={len(self)}, "
            f"n_features={self.X_general.shape[1]}, n_heads={self.y_heads.shape[1]})"
        )

    def save(self, output_dir: str):
        """Salva dataset + configuração completa"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.metadata.to_parquet(output_path / 'metadata_samples.parquet', index=False)
        self.X_general.to_parquet(output_path / 'X_general.parquet', index=False)
        self.X_recent.to_parquet(output_path / 'X_recent.parquet', index=False)
        self.y_heads.to_parquet(output_path / 'y_heads.parquet', index=False)
        self.y_daily.to_parquet(output_path / 'y_daily.parquet', index=False)
        np.save(output_path / 'weights.npy', self.weights)

        with open(output_path / 'features.json', 'w') as f:
            json.dump(self.features, f, indent=2)

        with open(output_path / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=2, default=str)

        print(f"✓ Dataset salvo em: {output_path}")
        print(f"  • Amostras: {len(self):,}")
        print(f"  • Features: {self.X_general.shape[1]}")
        print(f"  • Heads: {self.y_heads.shape[1]}")

    @classmethod
    def load(cls, dataset_dir: str) -> 'ProfileDataset':
        """Carrega dataset salvo com configuração"""
        path = Path(dataset_dir)

        config = {}
        config_path = path / 'config.json'
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)

        features = {}
        features_path = path / 'features.json'
        if features_path.exists():
            with open(features_path) as f:
                raw = json.load(f)
            features = {
                prefix: [(item[0], item[1]) for item in items]
                for prefix, items in raw.items()
            }

        ds = cls(
            metadata=pd.read_parquet(path / 'metadata_samples.parquet'),
            X_general=pd.read_parquet(path / 'X_general.parquet'),
            X_recent=pd.read_parquet(path / 'X_recent.parquet'),
            y_heads=pd.read_parquet(path / 'y_heads.parquet'),
            y_daily=pd.read_parquet(path / 'y_daily.parquet'),
            weights=np.load(path / 'weights.npy'),
            features=features,
            config=config,
        )

        print(f"✓ Dataset carregado: {dataset_dir}")
        print(f"  • Amostras: {len(ds):,}")
        print(f"  • Features: {ds.X_general.shape[1]}")
        print(f"  • X_recent: {ds.X_recent.shape}")
        print(f"  • Heads: {ds.y_heads.shape[1]}")

        return ds


# =====================================================================
# DataInput / GenerateDataInput
# =====================================================================

@dataclass
class DataInput:
    """Container para dados de entrada do predictor."""
    X_general: pd.DataFrame
    X_recent: pd.DataFrame
    features: Dict[str, List[Tuple[str, List[str]]]]
    upper: np.ndarray


class GenerateDataInput:
    """
    Gera :class:`DataInput` a partir de perfis expandidos e dados recentes.

    Descobre automaticamente o mapeamento canónico → expandido inspecionando
    as colunas de ``df_profile``: agrupa por prefixo e usa
    ``get_canonical_features`` de cada extractor.

    Parameters
    ----------
    target : str
        Métrica alvo (``'km'`` ou ``'h'``).
    df_profile : pd.DataFrame
        DataFrame com features expandidas (sufixo ``_{target}``)
        e, opcionalmente, colunas ``cluster_*``.
        Deve conter ``veiculo_id``.
    df_recent : pd.DataFrame
        DataFrame diário com colunas ``veiculo_id``, ``data`` e
        ``{target}_dia_clean``.
    upper : np.ndarray
        Valor upper por veículo (N,).
    """

    def __init__(
        self,
        target: str,
        df_profile: pd.DataFrame,
        df_recent: pd.DataFrame,
        upper: np.ndarray,
    ):
        self.target = target
        self.df_profile = df_profile
        self.df_recent = df_recent
        self.upper = np.asarray(upper, dtype=np.float64)

    def generate(self) -> DataInput:
        """Constrói :class:`DataInput` pronto para o predictor."""
        all_cols = [c for c in self.df_profile.columns if c != 'veiculo_id']

        # Separar cluster das features
        cluster_cols = sorted(c for c in all_cols if c.startswith('cluster_'))
        feature_cols = [c for c in all_cols if not c.startswith('cluster_')]

        # Validar sufixo _{target}
        suffix = f'_{self.target}'
        invalid = [c for c in feature_cols if not c.endswith(suffix)]
        if invalid:
            raise ValueError(
                f"Colunas não correspondem ao target '{self.target}': {invalid}"
            )

        # Agrupar por prefixo
        prefix_groups: Dict[str, List[str]] = {}
        for col in feature_cols:
            prefix = col.split('_')[0]
            prefix_groups.setdefault(prefix, []).append(col)

        # Descobrir canónicas via extractors
        features_map: Dict[str, List[Tuple[str, List[str]]]] = {}
        for prefix, cols in prefix_groups.items():
            extractor = get_extractor_by_prefix(
                prefix, features=None, metric=self.target,
            )
            canonical_map = extractor.get_canonical_map(cols)
            entries = [(k, v) for k, v in canonical_map.items()]
            if entries:
                features_map[prefix] = entries

        selected_cols = sorted(set(feature_cols + cluster_cols))
        X_general = self.df_profile[selected_cols].reset_index(drop=True)
        X_recent = self._build_X_recent()

        # Validação de nulos antes de devolver
        errors = []

        null_general = X_general.isnull().any()
        if null_general.any():
            bad_cols = null_general[null_general].index.tolist()
            for col in bad_cols:
                bad_rows = X_general[X_general[col].isnull()].index.tolist()
                vids = self.df_profile.loc[bad_rows, "veiculo_id"].tolist()
                logger.error(
                    "X_general NaN em '%s' para %d veículos: %s",
                    col, len(vids), vids[:20],
                )
            errors.append(f"X_general tem NaN em {len(bad_cols)} colunas: {bad_cols}")

        null_recent = X_recent.isnull().any()
        if null_recent.any():
            bad_cols = null_recent[null_recent].index.tolist()
            for col in bad_cols:
                bad_rows = X_recent[X_recent[col].isnull()].index.tolist()
                vids = self.df_profile.loc[bad_rows, "veiculo_id"].tolist()
                logger.error(
                    "X_recent NaN em '%s' para %d veículos: %s",
                    col, len(vids), vids[:20],
                )
            errors.append(f"X_recent tem NaN em {len(bad_cols)} colunas: {bad_cols}")

        if np.isnan(self.upper).any():
            bad_idx = np.where(np.isnan(self.upper))[0].tolist()
            vids = self.df_profile.loc[bad_idx, "veiculo_id"].tolist()
            logger.error("upper NaN para %d veículos: %s", len(vids), vids[:20])
            errors.append(f"upper tem NaN para {len(vids)} veículos: {vids[:20]}")

        if errors:
            raise ValueError(
                "DataInput contém valores nulos — impossível prosseguir com a predição.\n"
                + "\n".join(errors)
            )

        return DataInput(
            X_general=X_general,
            X_recent=X_recent,
            features=features_map,
            upper=self.upper,
        )

    def _build_X_recent(self) -> pd.DataFrame:
        target_col = f"{self.target}_dia_clean"
        vehicle_order = self.df_profile["veiculo_id"].values

        df = self.df_recent.sort_values(["veiculo_id", "data"])
        grouped = df.groupby("veiculo_id")[target_col].apply(list)

        rows = [grouped.get(vid, []) for vid in vehicle_order]
        X_recent = pd.DataFrame(rows)
        X_recent.columns = [f"h_{i}" for i in range(1, X_recent.shape[1] + 1)]
        return X_recent


class ProfileDatasetGenerator:
    """
    Gera dataset supervisionado (X, y) a partir de VehicleProfile,
    aplicando critérios explícitos de qualidade de amostra.
    """
    
    def __init__(
        self,
        vehicle_profile,
        df_daily: pd.DataFrame,
        min_weeks_general: int = 12,
        num_weeks_recent: int = 4,
        horizon_weeks: Union[int, List[int]] = 4,
        daily_horizon: int = 7,
        min_recent_active_days: int = 5,
        cache_dir: Optional[Union[str, Path]] = None,
        cluster_features: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        vehicle_profile : VehicleProfile
            Perfil versionado dos veículos
        df_daily : pd.DataFrame
            Dados diários com colunas: ['veiculo_id', 'data', '{metric}_dia_clean']
        min_weeks_general : int
            Número mínimo de semanas no período efetivo
        num_weeks_recent : int
            Número de semanas do histórico recente
        horizon_weeks : int ou List[int]
            Se int: cada semana vira uma head (ex.: 4 → [1,1,1,1], n_heads=4).
            Se lista: cada elemento define quantas semanas aquela head agrega.
            Exemplos:
                [1, 1, 2] → 3 heads (sem1, sem2, sem3+sem4)
                [2, 2]    → 2 heads (sem1+sem2, sem3+sem4)
                4         → 4 heads (sem1, sem2, sem3, sem4)
        daily_horizon : int
            Horizonte diário curto (ex.: 7)
        min_recent_active_days : int
            Mínimo de dias ativos no histórico recente
        cache_dir : str ou Path, optional
            Diretório para cache em disco. Se None, cache é desabilitado.
        cluster_features : str ou None
            Modo de inclusão de features de cluster:
            ``'prediction'`` — one-hot (cluster_0..cluster_n, valores 0/1)
            ``'probabilities'`` — probabilidades (cluster_0..cluster_n, valores 0–1)
            ``None`` — sem features de cluster.
        """
        self.vp = vehicle_profile
        self.df_daily = df_daily.copy()
        self.df_daily['data'] = pd.to_datetime(self.df_daily['data'])
        
        self.target = self.vp.target
        
        self.min_weeks_general = min_weeks_general
        self.num_weeks_recent = num_weeks_recent
        self.daily_horizon = daily_horizon
        self.min_recent_active_days = min_recent_active_days
        if cluster_features is not None and cluster_features not in ('prediction', 'probabilities'):
            raise ValueError(
                f"cluster_features deve ser 'prediction', 'probabilities' ou None, "
                f"recebeu {cluster_features!r}"
            )
        self.cluster_features = cluster_features
        
        # Normalizar horizon_weeks para lista
        if isinstance(horizon_weeks, int):
            self.head_weeks = [1] * horizon_weeks
        else:
            self.head_weeks = list(horizon_weeks)
        
        if not self.head_weeks or any(w < 1 for w in self.head_weeks):
            raise ValueError("horizon_weeks deve conter inteiros >= 1")
        
        self.n_heads = len(self.head_weeks)
        self.total_horizon_weeks = sum(self.head_weeks)
        
        # Dias por head (para agregação)
        self.head_days = [w * 7 for w in self.head_weeks]
        
        # Cache em disco
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

        predictions = self.vp.df_versions['cluster_prediction'].values
        proba_cols = sorted([
            c for c in self.vp.classification_columns
            if c.startswith('cluster_proba_')
        ])
        probabilities = (
            self.vp.df_versions[proba_cols].values if proba_cols else None
        )

        self.df_cluster = expand_cluster_columns(
            predictions=predictions,
            probabilities=probabilities,
            cluster_features=self.cluster_features,
            target=self.vp.metric,
        )
        
        # Validações
        if num_weeks_recent > min_weeks_general:
            raise ValueError("num_weeks_recent deve ser <= min_weeks_general")
        
        # Construir lista de feature columns para X_general
        self._build_feature_columns()
        
        # Cache de dados diários por veículo
        self._cache_daily_data()
    
    def _build_feature_columns(self):
        """Constrói lista de colunas para X_general com base na config."""
        prefix_to_extractor = {
            ext.PREFIX: ext for ext in self.vp.feature_extractors if ext.PREFIX is not None
        }
        self._features_map: Dict[str, List] = {}
        base_columns = []
        for prefix, base_list in self.vp.features.items():
            extractor = prefix_to_extractor.get(prefix)
            if extractor is None:
                continue
            entries = []
            for base_feat in base_list:
                expanded = extractor.get_expanded_features(base_feat)
                if expanded is not None:
                    if isinstance(expanded, str):
                        expanded = [expanded]
                    entries.append((base_feat, expanded))
                    base_columns.extend(expanded)
            if entries:
                self._features_map[prefix] = entries

        # Adicionar features de classificação (cluster)
        cluster_cols = list(self.df_cluster.columns)

        self._cluster_cols = cluster_cols
        self.feature_columns = sorted(base_columns + cluster_cols)

    def _cache_daily_data(self):
        """Cria cache de dados diários por veículo para acesso rápido"""
        self.daily_cache = {}
        for vid, group in self.df_daily.groupby('veiculo_id'):
            self.daily_cache[vid] = group.sort_values('data').set_index('data')[self.target]
    
    @property
    def config(self) -> dict:
        """Configuração completa do loader"""
        return {
            'metric': self.vp.metric,
            'min_weeks_general': self.min_weeks_general,
            'num_weeks_recent': self.num_weeks_recent,
            'head_weeks': self.head_weeks,
            'head_days': self.head_days,
            'n_heads': self.n_heads,
            'total_horizon_weeks': self.total_horizon_weeks,
            'daily_horizon': self.daily_horizon,
            'min_recent_active_days': self.min_recent_active_days,
            'feature_names': self.feature_columns,
            'cluster_features': self.cluster_features,
        }

    def _compute_data_hash(self) -> str:
        """Hash determinístico dos inputs de geração para cache."""
        h = hashlib.sha256()
        h.update(pd.util.hash_pandas_object(self.vp.df_versions).values.tobytes())
        h.update(pd.util.hash_pandas_object(self.vp.df_effective_period).values.tobytes())
        h.update(pd.util.hash_pandas_object(self.df_daily).values.tobytes())
        cfg = json.dumps(self.config, sort_keys=True, default=str).encode()
        h.update(cfg)
        return h.hexdigest()

    def _try_load_cache(self) -> Optional[ProfileDataset]:
        """Tenta carregar dataset do cache. Retorna None se inválido."""
        if self.cache_dir is None:
            return None

        hash_file = self.cache_dir / "hash.txt"
        if not (self.cache_dir.exists() and hash_file.exists()):
            return None

        cached_hash = hash_file.read_text().strip()
        if cached_hash != self._compute_data_hash():
            logger.info("Hash diferente — regenerando dataset.")
            return None

        logger.info("Cache válido encontrado — carregando dataset de %s", self.cache_dir)
        dataset = ProfileDataset.load(str(self.cache_dir))
        logger.info("Dataset carregado do cache: %d amostras", len(dataset))
        return dataset

    def _save_cache(self, dataset: ProfileDataset) -> None:
        """Salva dataset e hash no cache_dir."""
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        dataset.save(str(self.cache_dir))
        (self.cache_dir / "hash.txt").write_text(self._compute_data_hash())
        logger.info("Dataset salvo em cache: %s", self.cache_dir)

    def generate(self, verbose: bool = True) -> ProfileDataset:
        """
        Gera dataset de treino (ou carrega do cache, se disponível).
        
        Returns
        -------
        ProfileDataset
            Container com metadata, X_general, X_recent, y_heads, y_daily, weights e config.
            Suporta unpacking: metadata, X_general, X_recent, y_heads, y_daily, weights = dataset
        """
        cached = self._try_load_cache()
        if cached is not None:
            return cached
        
        records_X_general = []
        records_X_recent = []
        records_y_heads = []
        records_y_daily = []
        records_weight = []
        records_meta = []
        
        df_versions = self.vp.df_versions
        
        if not self.df_cluster.empty:
            df_versions = pd.concat([df_versions, self.df_cluster], axis=1)
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"GERANDO DATASET - {self.vp.metric.upper()}")
            print(f"{'='*80}")
            print(f"Versões disponíveis: {len(df_versions)}")
            print(f"Veículos únicos: {df_versions['veiculo_id'].nunique()}")
            print(f"Critérios de qualidade:")
            print(f"  • min_weeks_general: {self.min_weeks_general}")
            print(f"  • num_weeks_recent: {self.num_weeks_recent}")
            print(f"  • min_recent_active_days: {self.min_recent_active_days}")
            print(f"Horizonte:")
            print(f"  • head_weeks: {self.head_weeks} (total={self.total_horizon_weeks} semanas)")
            print(f"  • n_heads: {self.n_heads}")
            print(f"  • head_days: {self.head_days}")
            print()
        
        rejected_reasons = {
            'min_weeks': 0,
            'no_recent_data': 0,
            'min_active_days': 0,
            'no_future_data': 0
        }
        
        iterator = tqdm(df_versions.iterrows(), total=len(df_versions), disable=not verbose)
        
        for _, row in iterator:
            vid = row['veiculo_id']
            year = int(row['year'])
            week = int(row['week'])
            upper = float(row['upper'])
            
            # (1) Qualidade do perfil (H_g)
            if row['n_semanas_periodo'] < self.min_weeks_general:
                rejected_reasons['min_weeks'] += 1
                continue
            
            # Data de referência (domingo)
            ref_date = pd.to_datetime(f"{year}-W{week:02d}-7", format="%G-W%V-%u")
            
            # (2) Histórico recente diário (H_r)
            if vid not in self.daily_cache:
                rejected_reasons['no_recent_data'] += 1
                continue
            
            recent_start = ref_date - timedelta(days=7 * self.num_weeks_recent)
            recent_mask = (self.daily_cache[vid].index <= ref_date) & (self.daily_cache[vid].index > recent_start)
            recent_vals = self.daily_cache[vid][recent_mask].values
            
            if len(recent_vals) < 7 * self.num_weeks_recent or (recent_vals == 0).all():
                rejected_reasons['no_recent_data'] += 1
                continue
            
            q_recent_active_days = int((recent_vals > 0).sum())
            q_recent_activity_ratio = q_recent_active_days / len(recent_vals)
            
            if q_recent_active_days < self.min_recent_active_days:
                rejected_reasons['min_active_days'] += 1
                continue
            
            # (3) Horizonte futuro (F_p)
            horizon_days = 7 * self.total_horizon_weeks
            future_end = ref_date + timedelta(days=horizon_days)
            future_mask = (self.daily_cache[vid].index > ref_date) & (self.daily_cache[vid].index <= future_end)
            future_vals = self.daily_cache[vid][future_mask].values
            
            if len(future_vals) < horizon_days:
                rejected_reasons['no_future_data'] += 1
                continue
            
            if np.isnan(future_vals).any():
                rejected_reasons['no_future_data'] += 1
                continue
            
            # (4) Features X_general
            X_general = row[self.feature_columns].to_dict()
            
            # (4b) X_recent: últimos dias do daily_cache
            records_X_recent.append(recent_vals.tolist())
            
            # (5) Heads agregadas
            y_heads = {}
            offset = 0
            for i, days in enumerate(self.head_days):
                y_heads[f'head_{i+1}'] = future_vals[offset:offset + days].sum()
                offset += days
            # (6) Diário curto
            y_daily = {
                f'd{i+1}': future_vals[i]
                for i in range(self.daily_horizon)
            }
            
            # (7) Peso da amostra
            sample_weight = min(q_recent_activity_ratio / 0.5, 1.0)
            
            # Metadata
            meta = {
                'veiculo_id': vid,
                'ref_date': ref_date,
                'year': year,
                'week': week,
                'upper': upper,
                'n_semanas_periodo': row['n_semanas_periodo'],
                'cluster': row.get('cluster_prediction', -1),
                'q_recent_active_days': q_recent_active_days,
                'q_recent_activity_ratio': q_recent_activity_ratio
            }
            
            records_X_general.append(X_general)
            records_y_heads.append(y_heads)
            records_y_daily.append(y_daily)
            records_weight.append(sample_weight)
            records_meta.append(meta)
        
        if verbose:
            print(f"\n✓ Geração concluída")
            print(f"  • Amostras aceitas: {len(records_X_general):,}")
            print(f"  • Taxa de aceitação: {len(records_X_general)/len(df_versions)*100:.1f}%")
            print(f"\nRejeições por critério:")
            total_rejected = sum(rejected_reasons.values())
            for reason, count in rejected_reasons.items():
                pct = count / total_rejected * 100 if total_rejected > 0 else 0
                print(f"  • {reason}: {count:,} ({pct:.1f}%)")
            print(f"{'='*80}\n")
        
        df_X_general = pd.DataFrame(records_X_general)
        X_recent = np.array(records_X_recent)
        df_y_heads = pd.DataFrame(records_y_heads)
        df_y_daily = pd.DataFrame(records_y_daily)
        weights = np.array(records_weight)
        df_meta = pd.DataFrame(records_meta)

        df_metadata = df_meta[['veiculo_id', 'ref_date', 'year', 'week', 'upper', 'cluster']].reset_index(drop=True)

        cols_recent = [f'h_{i}' for i in range(1, self.num_weeks_recent * 7 + 1)]
        df_X_recent = pd.DataFrame(X_recent, columns=cols_recent)

        dataset = ProfileDataset(
            metadata=df_metadata,
            X_general=df_X_general,
            X_recent=df_X_recent,
            y_heads=df_y_heads,
            y_daily=df_y_daily,
            weights=weights,
            features=self._features_map,
            config=self.config,
        )

        self._save_cache(dataset)
        return dataset

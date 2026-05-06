import json
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

from moviasai.profiling.classification import SegmentationClassifier, TypeClassifier
from moviasai.profiling.feature_extraction import (
    BaseFeatureExtractor,
    TypeFeatureExtractor,
    get_extractor_by_prefix,
)
from moviasai.profiling.utils import compute_effective_period
from moviasai.data.data_quality import DataQualityEvaluator
from moviasai.data.utils import expand_cluster_columns


class VehicleProfile:
    """
    Perfil de veículos: extração de features e detecção de anomalias.
    """

    def __init__(
        self,
        type_features: List[str],
        km_features: Dict[str, List[str]],
        h_features: Dict[str, List[str]],
        filter_thresholds: Dict[str, float],
        quantiles: Dict[str, float],
        sample_size: Optional[int] = None,
        p_upper: int = 95,
        extract_metadata: bool = False,
    ):
        self.filter_thresholds = filter_thresholds
        self.quantiles = quantiles
        self.sample_size = sample_size
        self.p_upper = p_upper
        self.extract_metadata = extract_metadata

        # Extractors
        self.type_extractor = TypeFeatureExtractor(features=type_features)
        self.km_extractors = self._build_extractors('km', km_features)
        self.h_extractors = self._build_extractors('h', h_features)

        # State
        self.df_sampled: Optional[pl.DataFrame] = None
        self.df_features: Optional[pd.DataFrame] = None
        self.df_metadata_km: Optional[pd.DataFrame] = None
        self.df_metadata_h: Optional[pd.DataFrame] = None
        self.df_no_anomalies: Optional[pd.DataFrame] = None

    @staticmethod
    def _build_extractors(metric: str, features: Optional[Dict[str, List[str]]]) -> Dict[str, object]:
        """Cria dicionário {prefixo: extractor} para uma métrica."""
        if not features:
            return {}
        extractors = {}
        for prefix, feat_list in features.items():
            if feat_list:
                extractors[prefix] = get_extractor_by_prefix(prefix, features=feat_list, metric=metric)
        return extractors

    @classmethod
    def from_config(
        cls,
        segmentation_config,
        profile_config,
        data_quality_config,
        override_features: bool = False,
        extract_metadata: bool = False,
    ) -> "VehicleProfile":
        """
        Cria VehicleProfile a partir de configs.

        Parameters
        ----------
        segmentation_config : SegmentationConfig
            Configuração de segmentação (features).
        profile_config : VehicleProfileConfig
            Configuração do perfil (sample_size, p_upper).
        data_quality_config : DataQualityConfig
            Configuração de qualidade de dados (filter_thresholds, quantiles).
        override_features : bool
            Se True, usa km_features e h_features de profile_config
            em vez de segmentation_config.
        extract_metadata : bool
            Se True, calcula metadata (dt_inicio, dt_fim, upper) por veículo.
        """
        if override_features:
            km_features = profile_config.features.km.model_dump(exclude_none=True)
            h_features = profile_config.features.h.model_dump(exclude_none=True)
        else:
            km_features = segmentation_config.features.km.model_dump(exclude_none=True)
            h_features = segmentation_config.features.h.model_dump(exclude_none=True)

        return cls(
            type_features=segmentation_config.features.type,
            km_features=km_features,
            h_features=h_features,
            filter_thresholds=data_quality_config.thresholds.model_dump(),
            quantiles=data_quality_config.quantiles,
            sample_size=profile_config.profile.sample_size,
            p_upper=profile_config.profile.p_upper,
            extract_metadata=extract_metadata,
        )

    def _apply_sample_size(self, df: pl.DataFrame) -> pl.DataFrame:
        """Mantém apenas os sample_size registos mais recentes por veículo."""
        if self.sample_size is None:
            return df
        return (
            df.sort(['veiculo_id', 'data'])
            .group_by('veiculo_id')
            .tail(self.sample_size)
        )

    def _compute_metadata(self, df_full: pl.DataFrame, df_sampled: pl.DataFrame, target: str) -> pd.DataFrame:
        """Calcula histórico (dt_inicio, dt_fim) e percentil p_upper por veículo para um target."""
        col = f"{target}_dia_clean"
        p = self.p_upper / 100

        # Histórico: datas de primeira/última atividade (antes do sample_size)
        df_active = df_full.filter(pl.col(col) > 0)
        df_period = (
            df_active
            .group_by('veiculo_id')
            .agg([
                pl.col('data').min().alias('dt_inicio'),
                pl.col('data').max().alias('dt_fim'),
            ])
        )

        # Percentil p_upper (após sample_size, apenas dias ativos)
        df_upper = (
            df_sampled
            .filter(pl.col(col) > 0)
            .group_by('veiculo_id')
            .agg(
                pl.col(col).quantile(p).alias('upper'),
            )
        )

        return df_period.join(df_upper, on='veiculo_id', how='left').to_pandas()

    def feature_engineering(self, df: pl.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """Extrai e combina features de todos os extractors."""
        df_sampled = self._apply_sample_size(df)
        self.df_sampled = df_sampled

        # Metadata
        if self.extract_metadata:
            self.df_metadata_km = self._compute_metadata(df, df_sampled, 'km')
            self.df_metadata_h = self._compute_metadata(df, df_sampled, 'h')

        if verbose:
            print("=" * 80)
            print("GERANDO FEATURES")
            print("=" * 80)

        features_type = self.type_extractor.extract(df_sampled)
        features_pd = features_type

        for extractors in [self.km_extractors, self.h_extractors]:
            for extractor in extractors.values():
                df_ext = extractor.extract(df_sampled)
                features_pd = features_pd.merge(
                    df_ext, on="veiculo_id", how="left", suffixes=("", "_dup")
                )

        dup_cols = [c for c in features_pd.columns if c.endswith("_dup")]
        features_pd = features_pd.drop(columns=dup_cols)

        if verbose:
            print(f"Features totais: {len(features_pd.columns) - 1}")
            print(f"Veículos: {len(features_pd)}")
            print()

        self.df_features = features_pd
        return features_pd

    def detect_anomalies(
        self,
        df_daily: pl.DataFrame,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """Detecta outliers físicos e séries não-modeláveis."""
        evaluator = DataQualityEvaluator(
            filter_thresholds=self.filter_thresholds,
            type_extractor=self.type_extractor,
            km_extractor=self.km_extractors.get('seg'),
            h_extractor=self.h_extractors.get('seg'),
            quantiles=self.quantiles,
        )

        df_annotated, df_clean = evaluator.evaluate(
            df_daily=df_daily,
            df_features=self.df_features,
            output_dir=output_dir,
        )

        self.df_features = df_annotated
        self.df_no_anomalies = df_clean

        # Manter apenas veículos sem anomalias em df_sampled
        valid_ids = df_clean['veiculo_id'].unique().tolist()
        self.df_sampled = self.df_sampled.filter(pl.col('veiculo_id').is_in(valid_ids))

        return df_annotated

    def add_type(self, type_classifier: "TypeClassifier") -> pd.DataFrame:
        """Adiciona probabilidades de tipo ao df_no_anomalies.

        Parameters
        ----------
        type_classifier : TypeClassifier
            Classificador de tipo treinado.

        Returns
        -------
        pd.DataFrame
            df_no_anomalies com colunas type_1, type_2, ..., type_n.
        """
        feature_names = type_classifier.feature_names
        missing = [f for f in feature_names if f not in self.df_no_anomalies.columns]
        if missing:
            raise ValueError(f"Features ausentes em df_no_anomalies: {missing}")

        X = self.df_no_anomalies[feature_names].values
        _, probabilities = type_classifier.predict(X)

        if probabilities is not None:
            for i in range(probabilities.shape[1]):
                self.df_no_anomalies[f"type_{i + 1}"] = probabilities[:, i]

        return self.df_no_anomalies

    def add_segment(
        self,
        segment_classifier: "SegmentationClassifier",
        target: str,
        cluster_features: Optional[str] = 'probabilities',
    ) -> pd.DataFrame:
        """Adiciona colunas de segmento ao df_no_anomalies.

        Parameters
        ----------
        segment_classifier : SegmentationClassifier
            Classificador de segmento treinado.
        target : str
            'km' ou 'h'.
        cluster_features : str ou None
            ``'probabilities'`` — colunas com probabilidades por cluster.
            ``'prediction'`` — one-hot a partir da predição.
            ``None`` — não adiciona colunas de cluster.

        Returns
        -------
        pd.DataFrame
            df_no_anomalies com colunas cluster_0_{target}, ..., cluster_n_{target}.
        """
        if cluster_features is None:
            return self.df_no_anomalies

        feature_names = segment_classifier.feature_names
        suffix = f"_{target}"
        invalid = [f for f in feature_names if not f.endswith(suffix)]
        if invalid:
            raise ValueError(
                f"Features incompatíveis com target='{target}': {invalid}"
            )

        missing = [f for f in feature_names if f not in self.df_no_anomalies.columns]

        if missing:
            df_tmp = self._resolve_missing_features(missing, target)
            df_work = self.df_no_anomalies.merge(df_tmp, on='veiculo_id', how='left')
        else:
            df_work = self.df_no_anomalies

        X = df_work[feature_names].values
        predictions, probabilities = segment_classifier.predict(X)

        df_cluster = expand_cluster_columns(
            predictions=predictions,
            probabilities=probabilities,
            cluster_features=cluster_features,
            target=target,
        )

        for col in df_cluster.columns:
            self.df_no_anomalies[col] = df_cluster[col].values

        return self.df_no_anomalies

    def _resolve_missing_features(
        self,
        missing: List[str],
        target: str,
    ) -> pd.DataFrame:
        """Extrai features ausentes agrupando por prefixo do extractor.

        Utiliza ``self.df_sampled`` (guardado por ``feature_engineering``).

        Parameters
        ----------
        missing : List[str]
            Nomes expandidos das features em falta (e.g. ``['day_1_mean_km']``).
        target : str
            ``'km'`` ou ``'h'``.

        Returns
        -------
        pd.DataFrame
            DataFrame com ``veiculo_id`` e as colunas em falta.
        """
        if self.df_sampled is None:
            raise RuntimeError(
                "df_sampled não disponível. Execute feature_engineering antes de add_segment."
            )

        extractors = self.km_extractors if target == 'km' else self.h_extractors

        # Agrupar features por prefixo
        prefix_features: Dict[str, List[str]] = {}
        for feat in missing:
            prefix = feat.split('_')[0]
            prefix_features.setdefault(prefix, []).append(feat)

        result = self.df_no_anomalies[['veiculo_id']].copy()

        for prefix, expanded_feats in prefix_features.items():
            if prefix not in extractors:
                raise ValueError(
                    f"Nenhum extractor configurado para prefixo='{prefix}' "
                    f"(target='{target}'). Features: {expanded_feats}"
                )

            # Criar extractor completo (todas as features) para inverter nomes
            full_extractor = get_extractor_by_prefix(prefix, features=None, metric=target)
            canonical = full_extractor.get_canonical_features(expanded_feats)
            tmp_extractor = get_extractor_by_prefix(
                prefix, features=canonical, metric=target,
            )
            df_ext = tmp_extractor.extract(self.df_sampled)
            result = result.merge(df_ext, on='veiculo_id', how='left')

        return result[['veiculo_id'] + missing]

    def _enrich_metadata_quality(self):
        """Enriquece df_metadata_km/h com quality e quality_reason de df_features."""
        if not self.extract_metadata or self.df_features is None:
            return
        if 'quality' not in self.df_features.columns:
            return

        quality_df = self.df_features[['veiculo_id', 'quality', 'quality_reason']].copy()

        for attr in ('df_metadata_km', 'df_metadata_h'):
            df_meta = getattr(self, attr)
            if df_meta is None:
                continue
            # Remover colunas quality anteriores se existirem
            for col in ('quality', 'quality_reason'):
                if col in df_meta.columns:
                    df_meta = df_meta.drop(columns=[col])
            setattr(self, attr, df_meta.merge(quality_df, on='veiculo_id', how='left'))

    def run_pipeline(
        self,
        df: pl.DataFrame,
        type_classifier: "TypeClassifier",
        segment_classifier_km: "SegmentationClassifier",
        segment_classifier_h: "SegmentationClassifier",
        cluster_features: Optional[str] = 'probabilities',
        output_dir: Optional[Union[str, Path]] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Executa pipeline completo e devolve DataFrame em formato long.

        Sequência: feature_engineering → detect_anomalies → add_type → add_segment (km + h).

        Parameters
        ----------
        df : pl.DataFrame
            Dados diários brutos.
        type_classifier : TypeClassifier
            Classificador de tipo treinado.
        segment_classifier_km : SegmentationClassifier
            Classificador de segmento KM treinado.
        segment_classifier_h : SegmentationClassifier
            Classificador de segmento H treinado.
        cluster_features : str ou None
            Modo de expansão de clusters (``'probabilities'``, ``'prediction'`` ou ``None``).
        output_dir : str, Path ou None
            Directoria para artefactos de qualidade de dados.
        verbose : bool
            Imprimir progresso.

        Returns
        -------
        pd.DataFrame
            Colunas: ``veiculo_id``, ``feature``, ``feature_class``, ``valor``.
        """
        self.feature_engineering(df, verbose=verbose)
        self.detect_anomalies(self.df_sampled, output_dir=output_dir)
        self._enrich_metadata_quality()
        self.add_type(type_classifier)
        self.add_segment(segment_classifier_km, target='km', cluster_features=cluster_features)
        self.add_segment(segment_classifier_h, target='h', cluster_features=cluster_features)

        # Propagar type + cluster para veículos inválidos em df_features
        self._propagate_classification_to_invalid(
            type_classifier, segment_classifier_km, segment_classifier_h,
            cluster_features,
        )

        return self._to_long_format(self.df_no_anomalies)

    def _propagate_classification_to_invalid(
        self,
        type_classifier: "TypeClassifier",
        segment_classifier_km: "SegmentationClassifier",
        segment_classifier_h: "SegmentationClassifier",
        cluster_features: Optional[str],
    ) -> None:
        """Aplica type + segment aos veículos inválidos em df_features.

        Assim ``get_invalid_vehicles_long()`` devolve features completas
        (incluindo cluster_*) para veículos que possuem dados suficientes.
        Veículos que não podem ser classificados ficam com NaN nessas colunas
        e serão filtrados por ``get_invalid_vehicles_long()``.
        """
        if self.df_features is None or 'quality' not in self.df_features.columns:
            return

        # Garantir que colunas type_*/cluster_* existem em df_features (NaN por defeito).
        # Válidos já têm valores via df_no_anomalies; inválidos começam com NaN.
        new_cols = [c for c in self.df_no_anomalies.columns
                    if c.startswith(("type_", "cluster_")) and c not in self.df_features.columns]
        for col in new_cols:
            self.df_features[col] = np.nan

        # Copiar valores dos veículos válidos (que já foram classificados)
        valid_idx = self.df_features[self.df_features['quality'] == 0].index
        for col in [c for c in self.df_features.columns if c.startswith(("type_", "cluster_"))]:
            if col in self.df_no_anomalies.columns:
                # Alinhar por veiculo_id
                mapping = self.df_no_anomalies.set_index('veiculo_id')[col]
                self.df_features.loc[valid_idx, col] = (
                    self.df_features.loc[valid_idx, 'veiculo_id'].map(mapping).values
                )

        # Agora tentar classificar veículos inválidos
        df_inv = self.df_features[self.df_features['quality'] != 0].copy()
        if df_inv.empty:
            return

        # --- Type ---
        type_feats = type_classifier.feature_names
        if all(f in df_inv.columns for f in type_feats):
            X_type = df_inv[type_feats].values
            mask_ok = ~(np.isnan(X_type).any(axis=1) | np.isinf(X_type).any(axis=1))
            if mask_ok.any():
                _, probs = type_classifier.predict(X_type[mask_ok])
                if probs is not None:
                    for i in range(probs.shape[1]):
                        col = f"type_{i + 1}"
                        self.df_features.loc[df_inv.index[mask_ok], col] = probs[:, i]

        # --- Segment KM ---
        self._classify_invalid_segment(df_inv, segment_classifier_km, 'km', cluster_features)
        # --- Segment H ---
        self._classify_invalid_segment(df_inv, segment_classifier_h, 'h', cluster_features)

    def _classify_invalid_segment(
        self,
        df_inv: pd.DataFrame,
        segment_classifier: "SegmentationClassifier",
        target: str,
        cluster_features: Optional[str],
    ) -> None:
        """Classifica segmento para veículos inválidos e propaga a df_features."""
        if cluster_features is None:
            return

        feature_names = segment_classifier.feature_names
        if not all(f in df_inv.columns for f in feature_names):
            return

        X = df_inv[feature_names].values
        mask_ok = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1))
        if not mask_ok.any():
            return

        predictions, probabilities = segment_classifier.predict(X[mask_ok])
        df_cluster = expand_cluster_columns(
            predictions=predictions,
            probabilities=probabilities,
            cluster_features=cluster_features,
            target=target,
        )

        for col in df_cluster.columns:
            if col not in self.df_features.columns:
                self.df_features[col] = np.nan
            self.df_features.loc[df_inv.index[mask_ok], col] = df_cluster[col].values

    def _to_long_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Converte DataFrame wide em formato long (veiculo_id, feature, feature_class, valor)."""
        type_names = set(self.type_extractor.feature_names)
        km_names = {f for ext in self.km_extractors.values() for f in ext.feature_names}
        h_names = {f for ext in self.h_extractors.values() for f in ext.feature_names}

        type_prob_cols = {c for c in df.columns if c.startswith("type_")}
        cluster_cols = {c for c in df.columns if c.startswith("cluster_")}

        feature_cols = sorted(
            (type_names | km_names | h_names | type_prob_cols | cluster_cols)
            & set(df.columns)
        )

        df_long = df.melt(
            id_vars=['veiculo_id'],
            value_vars=feature_cols,
            var_name='feature',
            value_name='valor',
        )

        def _classify(feat: str) -> str:
            if feat in type_names or re.match(r'^type_\d+$', feat):
                return 'type'
            if feat in km_names:
                return 'km'
            if feat in h_names:
                return 'h'
            return feat.rsplit('_', 1)[-1]

        df_long['feature_class'] = df_long['feature'].map(_classify)
        return df_long[['veiculo_id', 'feature', 'feature_class', 'valor']]

    def get_invalid_vehicles_long(self) -> pd.DataFrame:
        """Retorna veículos não-válidos em formato long.

        Filtra ``df_features`` por ``quality != 0`` e exclui veículos
        com NaN ou Infinity em qualquer coluna numérica.

        Returns
        -------
        pd.DataFrame
            Colunas: ``veiculo_id``, ``feature``, ``feature_class``, ``valor``.
        """
        empty = pd.DataFrame(columns=['veiculo_id', 'feature', 'feature_class', 'valor'])

        if self.df_features is None or 'quality' not in self.df_features.columns:
            return empty

        df_invalid = self.df_features[self.df_features['quality'] != 0].copy()

        if df_invalid.empty:
            return empty

        # Excluir veículos com NaN ou Infinity em colunas numéricas
        numeric_cols = df_invalid.select_dtypes(include='number').columns.drop('veiculo_id', errors='ignore')
        has_bad = (
            df_invalid[numeric_cols].isna().any(axis=1)
            | np.isinf(df_invalid[numeric_cols]).any(axis=1)
        )
        df_invalid = df_invalid[~has_bad]

        if df_invalid.empty:
            return empty

        return self._to_long_format(df_invalid)

    def __repr__(self) -> str:
        n = len(self.df_features) if self.df_features is not None else 0
        return (
            f"VehicleProfile(n_vehicles={n}, "
            f"sample_size={self.sample_size}, "
            f"p_upper={self.p_upper})"
        )


class VersionedVehicleProfile:
    """
    Perfil completo de veículos com versionamento semanal
    
    Mantém dois DataFrames:
    1. df_versions: versões semanais de todas as features + classificação
    2. df_effective_period: período efetivo versionado
    """

    def __init__(
        self,
        metric: str,
        features: Dict[str, List[str]],
        classifier_features: List[str],
        metric_classifier: SegmentationClassifier,
        sample_size: int = 364,
        p_upper: int = 95,
        n_jobs: int = 4,
        batch_size: int = 10,
        last: bool = False,
        drop_duplicates: bool = True,
    ):
        """
        Inicializa o perfil de veículos
        
        Parameters
        -----------
        metric : str
            Métrica ('km' ou 'h')
        features : Dict[str, List[str]]
            Features canônicas por prefixo de extractor
            (ex: {'seg': ['media', 'cv'], 'day': ['mean']})
        classifier_features : List[str]
            Features canônicas (seg) do classificador
        metric_classifier : SegmentationClassifier
            Classificador de segmentos 
        sample_size : int, optional
            Número máximo de amostras (padrão: 364)
        p_upper : int, optional
            Percentil superior para clipping (padrão: 95)
        n_jobs : int, optional
            Número de threads paralelas (padrão: 4)
        batch_size : int, optional
            Tamanho do lote para processamento (padrão: 10)
        last : bool, optional
            Se True, gera apenas a última versão do perfil (padrão: False)
        drop_duplicates : bool, optional
            Se True, remove versões duplicadas por veículo quando o período
            efetivo (dt_inicio, dt_fim) não mudou (padrão: True)
        """
        self.metric = metric
        self.target = f'{metric}_dia_clean'
        self.sample_size = sample_size
        self.p_upper = p_upper
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.last = last
        self.drop_duplicates = drop_duplicates
        
        self.features = features
        self.classifier_features = classifier_features
        self.metric_classifier = metric_classifier
        
        self.feature_extractors = self._build_extractors()
        
        self.df_versions: Optional[pd.DataFrame] = None
        self.df_effective_period: Optional[pd.DataFrame] = None
        
        self._setup_feature_names()

    def _build_extractors(self) -> List[BaseFeatureExtractor]:
        """Cria extractors a partir dos prefixos e features configurados.
        
        O seg extractor é criado com a união das features do classifier
        e do perfil (sem duplicatas). Os restantes são criados directamente.
        """
        extractors = []
        
        # Seg: union de classifier + profile (deduplicated, classifier primeiro)
        seg_profile = self.features.get('seg', [])
        seg_union = list(dict.fromkeys(self.classifier_features + seg_profile))
        
        if seg_union:
            extractors.append(
                get_extractor_by_prefix('seg', features=seg_union, metric=self.metric)
            )
        
        # Restantes extractors do perfil
        for prefix, feat_list in self.features.items():
            if prefix == 'seg':
                continue
            if feat_list:
                extractors.append(
                    get_extractor_by_prefix(prefix, features=feat_list, metric=self.metric)
                )
        
        if not extractors:
            raise ValueError("Ao menos um extractor de features deve ser fornecido.")
        
        return extractors

    @classmethod
    def from_config(cls, config, segmentation_config, output_config, metric: str, last: bool = False, drop_duplicates: bool = True) -> "VersionedVehicleProfile":
        """
        Cria VehicleProfile a partir de configurações.

        Parameters
        ----------
        config : api.config.vehicle_profile_config.VehicleProfileConfig
            Configuração carregada do YAML.
        segmentation_config : api.config.segmentation_config.SegmentationConfig
            Configuração de segmentação (para features do classifier).
        output_config : api.config.output_config.OutputConfig
            Configuração de diretórios de saída.
        metric : str
            Métrica ('km' ou 'h').
        last : bool, optional
            Se True, gera apenas a última versão do perfil (padrão: False).
        drop_duplicates : bool, optional
            Se True, remove versões duplicadas por veículo quando o período
            efetivo não mudou (padrão: True).
        """
        classifier_dir = Path(output_config.models.classification) / "stage2"

        metric_classifier = SegmentationClassifier.load_model(
            classifier_dir / f"stage2_{metric}_BEST.onnx"
        )

        classifier_features = getattr(segmentation_config.features, metric)

        metric_features = getattr(config.features, metric)

        features = {}
        if metric_features.seg:
            features['seg'] = metric_features.seg
        if metric_features.day:
            features['day'] = metric_features.day
        if metric_features.phase:
            features['phase'] = metric_features.phase
        if metric_features.cycle:
            features['cycle'] = metric_features.cycle

        return cls(
            metric=metric,
            features=features,
            classifier_features=classifier_features.seg,
            metric_classifier=metric_classifier,
            sample_size=config.profile.sample_size,
            p_upper=config.profile.p_upper,
            n_jobs=config.profile.n_jobs,
            batch_size=config.profile.batch_size,
            last=last,
            drop_duplicates=drop_duplicates,
        )

    def _setup_feature_names(self):
        """Define nomes das features a partir dos extractors"""
        self.all_feature_columns = []
        self.extractor_features = {}
        
        for extractor in self.feature_extractors:
            features = extractor.feature_names.copy()
            self.all_feature_columns.extend(features)
            self.extractor_features[extractor.name] = features
        
        self.weekday_base_features = []
        for extractor in self.feature_extractors:
            if extractor.PREFIX == 'day':
                for feat_name in extractor.feature_names:
                    parts = feat_name.split('_')
                    if len(parts) >= 3 and parts[0] == 'day':
                        feat_type = '_'.join(parts[2:])
                        if feat_type.endswith(f'_{self.metric}'):
                            feat_type = feat_type[:-len(f'_{self.metric}')]
                        if feat_type not in self.weekday_base_features:
                            self.weekday_base_features.append(feat_type)
        
        self.scaler_columns = ['upper', 'n_semanas_periodo']
        self.classification_columns = ['cluster_prediction']
        
        n_classes = self.metric_classifier.metadata.get('n_classes', 0)
        if n_classes > 0:
            self.classification_columns.extend([
                f'cluster_proba_{i}' for i in range(n_classes)
            ])

    def get_profile_features(self) -> List[str]:
        """Retorna nomes expandidos apenas das features do perfil.

        Usa ``self.features`` (features canônicas por prefixo) e os
        extractors construídos para expandir cada feature base no seu
        nome final (com prefixo + sufixo).

        Returns
        -------
        List[str]
            Lista de nomes expandidos das features do perfil.
        """
        prefix_to_extractor = {
            ext.PREFIX: ext for ext in self.feature_extractors if ext.PREFIX is not None
        }

        features: List[str] = []
        for prefix, base_list in self.features.items():
            extractor = prefix_to_extractor.get(prefix)
            if extractor is None:
                continue
            for base_feat in base_list:
                expanded = extractor.get_expanded_features(base_feat)
                if expanded is None:
                    continue
                if isinstance(expanded, list):
                    features.extend(expanded)
                else:
                    features.append(expanded)

        return features

    @staticmethod
    def _get_week_end_date(date: Union[str, datetime]) -> datetime:
        """Retorna domingo"""
        if isinstance(date, str):
            date = pd.to_datetime(date, format='%d/%m/%Y')
        days_to_sunday = 6 - date.weekday()
        return date + timedelta(days=days_to_sunday)
    
    @staticmethod
    def _process_week_version(
        year: int,
        week: int,
        df_pl: pl.DataFrame,
        target: str,
        p_upper: int,
        sample_size: int,
        feature_extractors: List[BaseFeatureExtractor],
        metric_classifier: SegmentationClassifier,
        verbose: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, Tuple[float, List[float]]]]:
        """Processa uma semana"""
        df_week = df_pl.with_columns([
            pl.col('data').dt.iso_year().alias('y'),
            pl.col('data').dt.week().alias('w')
        ]).filter((pl.col('y') == year) & (pl.col('w') == week))
        
        if len(df_week) == 0:
            return None, None, {}
        
        week_end = df_week['data'].max()
        df_cumulative = df_pl.filter(pl.col('data') <= week_end)
        
        # Extrair features
        extracted_dfs = []
        for extractor in feature_extractors:
            df_features = extractor.extract(df_cumulative)
            extracted_dfs.append(df_features)
        
        df_combined = extracted_dfs[0].copy()
        
        # Classificação
        classifier_features = metric_classifier.feature_names
        missing_features = [f for f in classifier_features if f not in df_combined.columns]
        
        if missing_features:
            raise ValueError(f"Features faltando para classificação: {missing_features}")
        
        X_classify = df_combined[classifier_features].values
        predictions, probabilities = metric_classifier.predict(X_classify)
        
        df_combined['cluster_prediction'] = predictions
        
        if probabilities is not None:
            for i in range(probabilities.shape[1]):
                df_combined[f'cluster_proba_{i}'] = probabilities[:, i]
        else:
            n_classes = metric_classifier.metadata.get('n_classes', len(np.unique(predictions)))
            for i in range(n_classes):
                df_combined[f'cluster_proba_{i}'] = 0.0
        
        # Merge das outras features
        for df_features in extracted_dfs[1:]:
            df_combined = df_combined.merge(df_features, on='veiculo_id', how='left')
            for col in df_features.columns:
                if col != 'veiculo_id' and col in df_combined.columns:
                    df_combined[col] = df_combined[col].fillna(0)
        
        df_combined['year'] = year
        df_combined['week'] = week
        
        # Calcular período efetivo

        df_period = compute_effective_period(df_cumulative, target).to_pandas()
        df_period['year'] = year
        df_period['week'] = week
        
        # Fazer merge para ter dt_inicio e dt_fim disponíveis
        df_combined = df_combined.merge(
            df_period[['veiculo_id', 'dt_inicio', 'dt_fim']], 
            on='veiculo_id', 
            how='left'
        )
        
        # Calcular número de semanas
        df_combined['n_semanas_periodo'] = (
            (pd.to_datetime(df_combined['dt_fim']) - pd.to_datetime(df_combined['dt_inicio'])).dt.days / 7
        ).astype(int) + 1
        
        # Preencher NaN com 0 (caso não tenha período)
        df_combined['n_semanas_periodo'] = df_combined['n_semanas_periodo'].fillna(0).astype(int)
        
        # Remover dt_inicio e dt_fim do df_combined (já estão no df_period)
        df_combined = df_combined.drop(columns=['dt_inicio', 'dt_fim'])
        
        # Calcular scaler
        scaler_data = {}
        vehicle_iter = tqdm(df_combined['veiculo_id'], desc="Veículos", disable=not verbose)
        for vehicle_id in vehicle_iter:
            df_v = df_cumulative.filter(pl.col('veiculo_id') == vehicle_id).sort('data')
            values = df_v[target].to_numpy()[-sample_size:]
            upper = float(np.percentile(values, p_upper)) if len(values) > 0 else 1.0
            scaler_data[vehicle_id] = (upper, values.tolist())
        
        df_combined['upper'] = df_combined['veiculo_id'].map(lambda v: scaler_data[v][0])
        
        return df_combined, df_period, scaler_data
    
    def fit(self, df: Union[pl.DataFrame, pd.DataFrame], verbose: bool = True):
        """Ajusta perfis com paralelização otimizada"""
        if isinstance(df, pd.DataFrame):
            df_pl = pl.from_pandas(df)
        else:
            df_pl = df
        
        required_cols = ['veiculo_id', 'data', self.target]
        missing = [col for col in required_cols if col not in df_pl.columns]
        if missing:
            raise ValueError(f"Colunas faltando: {missing}")
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"AJUSTANDO PERFIS ({self.metric.upper()})")
            print(f"{'='*80}")
            print(f"Target: {self.target}")
            print(f"Veículos: {df_pl['veiculo_id'].n_unique()}")
            print(f"Período: {df_pl['data'].min()} a {df_pl['data'].max()}")
            print(f"Threads: {self.n_jobs}")
            print(f"Batch size: {self.batch_size}")
            print(f"Extractors: {len(self.feature_extractors)}")
            for extractor in self.feature_extractors:
                print(f"  • {extractor.name}: {len(extractor.feature_names)} features")
            print(f"Classificador: {self.metric_classifier.name}")
            print(f"  Features usadas: {len(self.metric_classifier.feature_names)}")
            print(f"  Classes: {self.metric_classifier.metadata.get('n_classes', 'N/A')}")
            print()
        
        df_with_week = df_pl.with_columns([
            pl.col('data').dt.iso_year().alias('year'),
            pl.col('data').dt.week().alias('week')
        ])
        
        all_weeks = (
            df_with_week
            .select(['year', 'week'])
            .unique()
            .sort(['year', 'week'])
        ).to_pandas()
        
        if self.last:
            all_weeks = all_weeks.tail(1)
        
        start_time = time.time()
        results = []
        
        if self.last:
            if verbose:
                row = all_weeks.iloc[0]
                print(f"📅 Processando semana {int(row['year'])}W{int(row['week']):02d}...")
            row = all_weeks.iloc[0]
            res = VersionedVehicleProfile._process_week_version(
                int(row['year']),
                int(row['week']),
                df_pl,
                self.target,
                self.p_upper,
                self.sample_size,
                self.feature_extractors,
                self.metric_classifier,
                verbose=verbose
            )
            if res[0] is not None:
                results.append(res)
        else:
            if verbose:
                print(f"📅 Total de semanas: {len(all_weeks)}")
                print("🔄 Processando em lotes paralelos...")
            
            n_batches = (len(all_weeks) + self.batch_size - 1) // self.batch_size
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(all_weeks))
                batch = all_weeks.iloc[start_idx:end_idx]
                with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                    futures = [
                        executor.submit(
                            VersionedVehicleProfile._process_week_version,
                            int(row['year']),
                            int(row['week']),
                            df_pl,
                            self.target,
                            self.p_upper,
                            self.sample_size,
                            self.feature_extractors,
                            self.metric_classifier
                        )
                        for _, row in batch.iterrows()
                    ]
                    
                    desc = f"Batch {batch_idx+1}/{n_batches}" if verbose else None
                    for f in tqdm(as_completed(futures), total=len(futures), desc=desc, disable=not verbose):
                        res = f.result()
                        if res[0] is not None:
                            results.append(res)
        
        valid_results = results
        all_versions = [r[0] for r in valid_results]
        all_periods = [r[1] for r in valid_results]
        
        self.df_versions = pd.concat(all_versions, ignore_index=True)
        self.df_effective_period = pd.concat(all_periods, ignore_index=True)
        
        base_cols = ['year', 'week', 'veiculo_id']
        feature_cols = (
            self.all_feature_columns +
            self.scaler_columns +
            self.classification_columns
        )
        
        self.df_versions = self.df_versions[base_cols + feature_cols]
        
        period_cols = ['year', 'week', 'veiculo_id', 'dt_inicio', 'dt_fim']
        self.df_effective_period = self.df_effective_period[period_cols]
        
        # Remover versões duplicadas (mesmo veículo, mesmo período efetivo)
        if self.drop_duplicates:
            n_before = len(self.df_versions)
            ep = self.df_effective_period.sort_values(
                ['veiculo_id', 'year', 'week']
            ).reset_index(drop=True)
            
            shifted_inicio = ep.groupby('veiculo_id')['dt_inicio'].shift(1)
            shifted_fim = ep.groupby('veiculo_id')['dt_fim'].shift(1)
            
            first = ~ep.duplicated(subset=['veiculo_id'], keep='first')
            changed = first | (ep['dt_inicio'] != shifted_inicio) | (ep['dt_fim'] != shifted_fim)
            
            keep_keys = ep.loc[changed, ['year', 'week', 'veiculo_id']]
            
            self.df_effective_period = self.df_effective_period.merge(
                keep_keys, on=['year', 'week', 'veiculo_id']
            )
            self.df_versions = self.df_versions.merge(
                keep_keys, on=['year', 'week', 'veiculo_id']
            )
            
            n_after = len(self.df_versions)
            n_dropped = n_before - n_after
        
        elapsed = time.time() - start_time
        
        if verbose:
            print(f"✓ Concluído em {elapsed:.2f}s")
            print(f"  - Versões: {len(self.df_versions)}")
            if self.drop_duplicates:
                print(f"  - Duplicadas removidas: {n_dropped:,}")
            print(f"  - Veículos: {self.df_versions['veiculo_id'].nunique()}")
            
            if 'cluster_prediction' in self.df_versions.columns:
                cluster_counts = self.df_versions['cluster_prediction'].value_counts()
                print(f"  - Distribuição de clusters:")
                for cluster, count in cluster_counts.items():
                    pct = count / len(self.df_versions) * 100
                    print(f"      Cluster {cluster}: {count:,} ({pct:.1f}%)")
            
            print(f"{'='*80}\n")

    def get_profile(
        self,
        vehicle_id: Union[int, List[int]],
        date: Union[str, datetime],
        normalized: bool = False
    ) -> pd.DataFrame:
        """Retorna perfil(s) válido(s) para uma data"""
        if isinstance(date, str):
            date = pd.to_datetime(date, format='%d/%m/%Y')
        
        week_end = self._get_week_end_date(date)
        target_year = week_end.year
        target_week = week_end.isocalendar()[1]
        
        if isinstance(vehicle_id, int):
            vehicle_ids = [vehicle_id]
        else:
            vehicle_ids = vehicle_id
        
        results = []
        
        for vid in vehicle_ids:
            df_v = self.df_versions[self.df_versions['veiculo_id'] == vid]
            
            if len(df_v) == 0:
                warnings.warn(f"Veículo {vid} não encontrado")
                continue
            
            exact = df_v[(df_v['year'] == target_year) & (df_v['week'] == target_week)]
            
            if len(exact) > 0:
                row = exact.iloc[0]
            else:
                prior = df_v[
                    (df_v['year'] < target_year) |
                    ((df_v['year'] == target_year) & (df_v['week'] < target_week))
                ].sort_values(['year', 'week'], ascending=False)
                
                if len(prior) > 0:
                    row = prior.iloc[0]
                else:
                    warnings.warn(f"Nenhuma versão para veículo {vid}")
                    continue
            
            if normalized:
                row = row.copy()
                upper = row['upper']
                if self.feature_extractors:
                    seg_features = self.extractor_features.get(
                        self.feature_extractors[0].name, []
                    )
                    for feat in seg_features:
                        if feat in row:
                            row[feat] = np.clip(row[feat], 0, upper) / upper if upper > 0 else 0
            
            results.append(row)
        
        return pd.DataFrame(results).reset_index(drop=True) if results else pd.DataFrame()

    def get_profiles_batch(
        self,
        vehicle_ids: List[int],
        dates: List[Union[str, datetime]],
        normalized: bool = False
    ) -> Dict[Tuple[int, datetime], pd.Series]:
        """Busca perfis em BATCH"""
        dates_dt = []
        for date in dates:
            if isinstance(date, str):
                dates_dt.append(pd.to_datetime(date, format='%d/%m/%Y'))
            else:
                dates_dt.append(date)
        
        date_to_week = {}
        for date in dates_dt:
            week_end = self._get_week_end_date(date)
            year = week_end.year
            week = week_end.isocalendar()[1]
            date_to_week[date] = (year, week)
        
        unique_weeks = list(set(date_to_week.values()))
        
        year_week_filters = []
        for year, week in unique_weeks:
            year_week_filters.append(
                (self.df_versions['year'] == year) & 
                (self.df_versions['week'] == week)
            )
        
        if year_week_filters:
            combined_filter = year_week_filters[0]
            for f in year_week_filters[1:]:
                combined_filter = combined_filter | f
            
            df_filtered = self.df_versions[
                combined_filter & 
                self.df_versions['veiculo_id'].isin(vehicle_ids)
            ].copy()
        else:
            df_filtered = pd.DataFrame()
        
        profiles_cache = {}
        
        for date in dates_dt:
            target_year, target_week = date_to_week[date]
            
            for vid in vehicle_ids:
                exact = df_filtered[
                    (df_filtered['veiculo_id'] == vid) &
                    (df_filtered['year'] == target_year) &
                    (df_filtered['week'] == target_week)
                ]
                
                if len(exact) > 0:
                    row = exact.iloc[0].copy()
                else:
                    prior = df_filtered[
                        (df_filtered['veiculo_id'] == vid) &
                        (
                            (df_filtered['year'] < target_year) |
                            ((df_filtered['year'] == target_year) & (df_filtered['week'] < target_week))
                        )
                    ].sort_values(['year', 'week'], ascending=False)
                    
                    if len(prior) > 0:
                        row = prior.iloc[0].copy()
                    else:
                        continue
                
                if normalized:
                    upper = row['upper']
                    if self.feature_extractors:
                        seg_features = self.extractor_features.get(
                            self.feature_extractors[0].name, []
                        )
                        for feat in seg_features:
                            if feat in row:
                                row[feat] = np.clip(row[feat], 0, upper) / upper if upper > 0 else 0
                
                profiles_cache[(vid, date)] = row
        
        return profiles_cache

    def save(self, output_dir: str):
        """Salva perfis"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.df_versions.to_parquet(output_path / 'vehicle_profiles.parquet', index=False)
        self.df_effective_period.to_parquet(output_path / 'effective_periods.parquet', index=False)
        
        extractors_config = []
        for extractor in self.feature_extractors:
            config = {
                'name': extractor.name,
                'type': extractor.__class__.__name__,
                'base_features': extractor.selected_features,
                'features': extractor.feature_names
            }
            
            if hasattr(extractor, 'metric'):
                config['metric'] = extractor.metric
            if hasattr(extractor, 'add_suffix'):
                config['add_suffix'] = extractor.add_suffix
            
            extractors_config.append(config)
        
        config_data = {
            'features': self.features,
            'classifier_features': self.classifier_features,
            'extractors': extractors_config,
            'classifier': {
                'name': self.metric_classifier.name,
                'features': self.metric_classifier.feature_names,
                'model_name': self.metric_classifier.best_model_name,
                'n_classes': self.metric_classifier.metadata.get('n_classes', 0)
            }
        }
        
        with open(output_path / 'extractor_config.json', 'w') as f:
            json.dump(config_data, f, indent=2)
        
        metadata = {
            'target': self.target,
            'metric': self.metric,
            'sample_size': self.sample_size,
            'p_upper': self.p_upper,
            'n_vehicles': int(self.df_versions['veiculo_id'].nunique()),
            'n_versions': len(self.df_versions),
            'n_extractors': len(self.feature_extractors),
            'total_features': len(self.all_feature_columns),
            'n_classification_columns': len(self.classification_columns),
            'classifier_name': self.metric_classifier.name,
            'avg_semanas_periodo': float(self.df_versions['n_semanas_periodo'].mean()),
            'min_semanas_periodo': int(self.df_versions['n_semanas_periodo'].min()),
            'max_semanas_periodo': int(self.df_versions['n_semanas_periodo'].max())
        }
        
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, profile_dir: str, classifier_path: str) -> 'VersionedVehicleProfile':
        """Carrega perfis"""
        output_path = Path(profile_dir)
        
        with open(output_path / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        with open(output_path / 'extractor_config.json', 'r') as f:
            config_data = json.load(f)
        
        metric_classifier = SegmentationClassifier.load_model(classifier_path)
        
        features = config_data.get('features', {})
        classifier_features = config_data.get('classifier_features', [])
        
        instance = cls(
            metric=metadata['metric'],
            features=features,
            classifier_features=classifier_features,
            metric_classifier=metric_classifier,
            sample_size=metadata['sample_size'],
            p_upper=metadata['p_upper']
        )
        
        instance.df_versions = pd.read_parquet(output_path / 'vehicle_profiles.parquet')
        instance.df_effective_period = pd.read_parquet(output_path / 'effective_periods.parquet')
        
        return instance

    def __repr__(self) -> str:
        n_versions = len(self.df_versions) if self.df_versions is not None else 0
        n_vehicles = self.df_versions['veiculo_id'].nunique() if self.df_versions is not None else 0
        
        return (
            f"VehicleProfile(\n"
            f"  metric='{self.metric}',\n"
            f"  n_vehicles={n_vehicles},\n"
            f"  n_versions={n_versions},\n"
            f"  n_extractors={len(self.feature_extractors)},\n"
            f"  total_features={len(self.all_feature_columns)},\n"
            f"  classifier='{self.metric_classifier.name}',\n"
            f"  n_jobs={self.n_jobs},\n"
            f"  batch_size={self.batch_size}\n"
            f")"
        )

import pandas as pd
import numpy as np
import json
import polars as pl
from enum import IntEnum
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from moviasai.profiling.feature_extraction import (
    BaseFeatureExtractor, SegmentationFeatureExtractor, TypeFeatureExtractor
)
from moviasai.profiling.utils import WeeklyActivitySlicer, merge_datasets


class SeriesQuality(IntEnum):
    """Classificação de qualidade de uma série temporal."""
    VALID = 0
    OUTLIER = 1
    NOT_MODELABLE = 2
    EMPTY = 3
    SINGLE_TARGET = 4


class SeriesQualityFilter:
    """
    Filtro de qualidade para séries temporais de veículos.
    
    Aplica limiares mínimos de qualidade (semanas ativas, gaps máximos)
    para identificar séries válidas e preencher datas ausentes.
    
    Attributes:
    -----------
    df : polars.DataFrame
        DataFrame com dados brutos
    target : str
        Target: ``'km'`` ou ``'h'``
    col_target : str
        Nome da coluna alvo (``f'{target}_dia_clean'``)
    min_weeks : int
        Número mínimo de semanas para veículos válidos
    max_gap : int
        Máximo de dias de gap permitido entre registros
    """
    
    def __init__(
        self, 
        df: pd.DataFrame | pl.DataFrame,
        target: str,
        min_weeks: int = 5, 
        max_gap: int = 3
    ):
        """
        Inicializa o filtro de qualidade
        
        Parameters:
        -----------
        df : pandas.DataFrame ou polars.DataFrame
            DataFrame com dados brutos.
        target : str
            Target: ``'km'`` ou ``'h'``
        min_weeks : int, optional
            Número mínimo de semanas para veículos válidos (padrão: 5)
        max_gap : int, optional
            Máximo de dias de gap permitido (padrão: 3). Deve ser >= 1.
            
        Examples:
        ---------
        >>> f = SeriesQualityFilter(df=df_polars, target='km')
        >>> f = SeriesQualityFilter(df=df_pandas, target='h', max_gap=7)
        
        Raises:
        -------
        ValueError
            Se as colunas obrigatórias não existirem no DataFrame
        """
        
        col_target = f"{target}_dia_clean"

        if isinstance(df, pd.DataFrame):
            required_cols = ['veiculo_id', 'data', col_target]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                raise ValueError(
                    f"Colunas obrigatórias ausentes no DataFrame: {missing_cols}. "
                    f"Colunas disponíveis: {list(df.columns)}"
                )
            
            df_subset = df[required_cols].copy()
            
            if not pd.api.types.is_datetime64_any_dtype(df_subset['data']):
                df_subset['data'] = pd.to_datetime(df_subset['data'])
            
            self.df = pl.from_pandas(df_subset)
                
        elif isinstance(df, pl.DataFrame):
            required_cols = ['veiculo_id', 'data', col_target]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                raise ValueError(
                    f"Colunas obrigatórias ausentes no DataFrame: {missing_cols}. "
                    f"Colunas disponíveis: {list(df.columns)}"
                )
            
            self.df = df.select(required_cols)
            
            if self.df['data'].dtype not in [pl.Date, pl.Datetime]:
                self.df = self.df.with_columns([
                    pl.col('data').str.strptime(pl.Date, format='%Y-%m-%d')
                ])
                
        else:
            raise ValueError(
                f"Tipo inválido para 'df': {type(df)}. "
                "Deve ser pandas.DataFrame ou polars.DataFrame"
            )
        
        # Armazenar parâmetros
        self.target = target
        self.col_target = col_target
        self.min_weeks = min_weeks
        self.max_gap = max_gap
        
        # Informações do dataset
        print(f"Dataset carregado: {len(self.df):,} registros")
        print(f"Veiculos: {self.df['veiculo_id'].n_unique()}")
        print(f"Target: {target}")
        print()

    def _fill_missing_dates(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Preenche datas ausentes com zero 
        
        Para cada veículo, cria uma sequência completa de datas entre
        min_date GLOBAL e max_date GLOBAL e preenche valores ausentes com 0.
        
        Isso garante que todos os veículos tenham séries do mesmo tamanho,
        facilitando o treinamento de modelos.
        
        Parameters:
        -----------
        df : polars.DataFrame
            DataFrame com janelas de dados
            
        Returns:
        --------
        polars.DataFrame
            DataFrame com datas preenchidas (todos os veículos com mesmo range)
        """
        # 1. Obter min e max data GLOBAL (de todos os registros)
        min_date_global = df['data'].min()
        max_date_global = df['data'].max()
        
        print(f"Preenchendo datas ausentes:")
        print(f"  • Range global: {min_date_global} a {max_date_global}")
        print(f"  • Total de dias: {(max_date_global - min_date_global).days + 1}")
        
        # 2. Gerar sequência completa de datas
        all_dates = pl.date_range(
            start=min_date_global,
            end=max_date_global,
            interval="1d",
            eager=True
        ).alias('data')
        
        # 3. Obter lista de todos os veículos
        all_vehicles = df['veiculo_id'].unique().sort()
        
        print(f"  • Veículos: {len(all_vehicles)}")
        
        # 4. Criar produto cartesiano: todos os veículos x todas as datas
        df_all_dates = pl.DataFrame({'data': all_dates})
        df_all_vehicles = pl.DataFrame({'veiculo_id': all_vehicles})
        
        # Cross join (produto cartesiano)
        df_complete = df_all_vehicles.join(df_all_dates, how='cross')
        
        print(f"  • Combinações totais: {len(df_complete):,}")
        
        # 5. Join left com df_window original
        df_filled = df_complete.join(
            df, 
            on=['veiculo_id', 'data'], 
            how='left'
        )
        
        # 6. Preencher valores ausentes com zero
        df_filled = df_filled.with_columns([
            pl.col(self.col_target).fill_null(0)
        ])
        
        # 7. Ordenar
        df_filled = df_filled.sort(['veiculo_id', 'data'])
        
        # Estatísticas
        n_zeros = (df_filled[self.col_target] == 0).sum()
        pct_zeros = n_zeros / len(df_filled) * 100
        
        print(f"  • Registros com zero: {n_zeros:,} ({pct_zeros:.1f}%)")
        print()
        
        return df_filled

    def filter(
        self, 
        verbose: bool = True
    ) -> pl.DataFrame:
        """
        Filtra o dataset aplicando limiares de qualidade e preenchendo datas.
        
        Pipeline:
        1. Constrói janelas de dados contínuos
        2. Preenche datas ausentes com zero
        
        Parameters:
        -----------
        verbose : bool, optional
            Se True, imprime informações (padrão: True)
            
        Returns:
        --------
        polars.DataFrame
            DataFrame filtrado com colunas:
            - veiculo_id: ID do veículo
            - data: Data do registro
            - {target}: Valor do target (0 para datas ausentes)

        Examples:
        ---------
        >>> f = SeriesQualityFilter(df=df, target='km')
        >>> df_filtered = f.filter()
        """
        if verbose:
            print("=" * 80)
            print("FILTRANDO DATASET")
            print("=" * 80)
            print(f"Target: {self.target}")
            print(f"Min weeks: {self.min_weeks}")
            print(f"Max gap: {self.max_gap}")
            print()
        
        builder = WeeklyActivitySlicer(target=self.target, stride=7, max_gap=self.max_gap, window_size=7)
        
        # 1. Construir janelas
        df_window, _ = builder.slice(
            self.df, 
            verbose=verbose
        )

        if len(df_window) == 0:
            return pl.DataFrame()
        
        if verbose:
            print(f"Apos construcao de janelas:")
            print(f"  • Registros: {len(df_window):,}")
            print(f"  • Veiculos: {df_window['veiculo_id'].n_unique()}")
            print(f"  • Segmentos: {df_window['window'].n_unique()}")
            print()

        
        # 2. Preencher datas ausentes com zero
        if verbose:
            print("Preenchendo datas ausentes...")
        
        df_window_final = self._fill_missing_dates(df_window).select(["veiculo_id", "data", self.col_target])
        
        if verbose:
            n_zeros = (df_window_final[self.col_target] == 0).sum()
            pct_zeros = n_zeros / len(df_window_final) * 100
            
            print(f"Apos preenchimento:")
            print(f"  • Registros totais: {len(df_window_final):,}")
            print(f"  • Registros com zero: {n_zeros:,} ({pct_zeros:.1f}%)")
            print(f"  • Veiculos: {df_window_final['veiculo_id'].n_unique()}")
            print()
        
        if verbose:
            print("=" * 80)
            print("DATASET FILTRADO COM SUCESSO")
            print("=" * 80)
            print()
        
        return df_window_final


class DataQualityEvaluator:
    """
    Detector de anomalias e avaliador de qualidade de dados.

    Uso principal: ``detect()`` executa o fluxo completo
    (formatação → fit → avaliação → persistência).
    """

    def __init__(
        self,
        filter_thresholds: Dict[str, float],
        quantiles: Dict[str, float],
        type_extractor: TypeFeatureExtractor = None,
        km_extractor: SegmentationFeatureExtractor = None,
        h_extractor: SegmentationFeatureExtractor = None,
    ):
        self.filter_thresholds = filter_thresholds
        self.quantiles = quantiles
        self.type_extractor = type_extractor
        self.km_extractor = km_extractor
        self.h_extractor = h_extractor

        self.computed_thresholds: Optional[dict] = None

    @classmethod
    def from_config(
        cls,
        config,
        **kwargs,
    ) -> "DataQualityEvaluator":
        """Instancia a partir de ``DataQualityConfig``.

        Parameters
        ----------
        config : api.config.data_quality_config.DataQualityConfig
            Configuração carregada do YAML.
        **kwargs
            Parâmetros adicionais (``type_extractor``, etc.).
        """
        return cls(
            filter_thresholds=config.thresholds.model_dump(),
            quantiles=config.quantiles,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def evaluate(
        self,
        df_daily: pl.DataFrame,
        df_features: pd.DataFrame,
        output_dir: Optional[Union[str, Path]] = None,
        verbose: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Executa detecção completa de anomalias: formatação, fit, avaliação e persistência.

        Parameters
        ----------
        df_daily : pl.DataFrame
            Dados diários brutos de telemetria.
        df_features : pd.DataFrame
            Features já extraídas (deve conter ``veiculo_id``).
        output_dir : str, Path ou None
            Diretório base para salvar artefactos de anomalias.
            Se None, não salva artefactos.
        verbose : bool
            Se True, imprime progresso.

        Returns
        -------
        (df_annotated, df_clean)
            ``df_annotated``: ``df_features`` com colunas de qualidade adicionadas.
            ``df_clean``: subconjunto válido (sem outliers nem séries não-modeláveis).
        """

        output_dir = Path(output_dir) if output_dir else None

        # 0. Identificar veículos vazios e single-target
        df_daily, empty_ids, km_only_ids, h_only_ids = self._detect_empty_series(
            df_daily, verbose=verbose,
        )
        single_target_ids = km_only_ids | h_only_ids

        # 1. Formatação
        formatter_km = SeriesQualityFilter(df=df_daily, target="km", **self.filter_thresholds)
        formatter_h = SeriesQualityFilter(df=df_daily, target="h", **self.filter_thresholds)
        df_formatted = merge_datasets(
            df_raw=df_daily,
            df_km=formatter_km.filter(),
            df_h=formatter_h.filter(),
        )

        all_ids = set(df_daily["veiculo_id"].unique().to_list()) | empty_ids
        formatted_ids = set(df_formatted["veiculo_id"].unique().to_list())
        failed_ids = (all_ids - formatted_ids) - empty_ids

        # Single-target vehicles que falharam na formatação de ambos os targets
        # são genuinamente não-modeláveis; os que passaram no target válido continuam
        truly_failed_ids = failed_ids - single_target_ids
        # Single-target que falharam: verificar se passaram no target disponível
        single_failed_ids = failed_ids & single_target_ids

        if verbose:
            print(f"\n📊 Análise de formatação:")
            print(f"  • Veículos totais: {len(all_ids):,}")
            print(f"  • Veículos formatados: {len(formatted_ids):,}")
            print(f"  • Veículos que falharam: {len(failed_ids):,}")
            if single_target_ids:
                print(f"  • Veículos single-target: {len(single_target_ids):,}")

        min_weeks = self.filter_thresholds["min_weeks"]
        max_gap = self.filter_thresholds["max_gap"]

        # 2. Marcar veículos com séries vazias + que falharam na formatação
        excluded_results = []
        if empty_ids:
            excluded_results.append(pd.DataFrame({
                "veiculo_id": list(empty_ids),
                "quality": SeriesQuality.EMPTY,
                "quality_reason": "Série vazia (km=0 e h=0)",
            }))
        if truly_failed_ids:
            reason = f"Série com min_weeks < {min_weeks} ou max_gap > {max_gap}"
            excluded_results.append(pd.DataFrame({
                "veiculo_id": list(truly_failed_ids),
                "quality": SeriesQuality.NOT_MODELABLE,
                "quality_reason": reason,
            }))
        if single_failed_ids:
            reason = f"Single-target: falhou min_weeks < {min_weeks} ou max_gap > {max_gap}"
            excluded_results.append(pd.DataFrame({
                "veiculo_id": list(single_failed_ids),
                "quality": SeriesQuality.NOT_MODELABLE,
                "quality_reason": reason,
            }))
        df_failed_results = pd.concat(excluded_results) if excluded_results else pd.DataFrame()

        if verbose and len(df_failed_results):
            print(f"  • Marcados como não-modeláveis/vazios: {len(df_failed_results):,}")

        if verbose:
            print()

        # 3. Fit + avaliação nos restantes
        excluded_ids = truly_failed_ids | single_failed_ids | empty_ids
        mask_not_failed = ~df_features["veiculo_id"].isin(excluded_ids)
        self._compute_thresholds(
            df_features[mask_not_failed & ~df_features["veiculo_id"].isin(single_target_ids)],
            verbose=verbose,
        )
        df_not_failed_results = self._evaluate_batch(
            df_features[mask_not_failed],
            single_target_ids=single_target_ids,
            verbose=verbose,
        )

        # 4. Combinar resultados
        df_results = pd.concat([df_failed_results, df_not_failed_results])
        df_annotated = df_features.merge(df_results, on="veiculo_id")

        # Adicionar coluna available_targets para todos os veículos
        def _available_targets(vid):
            if vid in km_only_ids:
                return "km"
            if vid in h_only_ids:
                return "h"
            if vid in empty_ids:
                return ""
            return "km,h"

        df_annotated["available_targets"] = df_annotated["veiculo_id"].map(_available_targets)

        # 5. Persistência
        if output_dir:
            self._save_artifacts(
                df_annotated, output_dir, all_ids, formatted_ids,
                truly_failed_ids | single_failed_ids,
            )

        # 6. Separar válidos (VALID + SINGLE_TARGET participam do pipeline)
        df_clean = df_annotated[
            df_annotated["quality"].isin([SeriesQuality.VALID, SeriesQuality.SINGLE_TARGET])
        ].copy()

        return df_annotated, df_clean

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_empty_series(
        df_daily: pl.DataFrame, verbose: bool = True,
    ) -> Tuple[pl.DataFrame, set, set, set]:
        """
        Identifica veículos com séries vazias e single-target.

        Veículos com ambos os targets a zero são removidos.
        Veículos com apenas um target válido são mantidos e sinalizados.

        Returns
        -------
        (df_filtered, empty_ids, km_only_ids, h_only_ids)
            ``df_filtered``: ``df_daily`` sem os veículos totalmente vazios.
            ``empty_ids``: ``veiculo_id`` com km=0 E h=0.
            ``km_only_ids``: ``veiculo_id`` com km>0 mas h=0.
            ``h_only_ids``: ``veiculo_id`` com h>0 mas km=0.
        """
        df_max = (
            df_daily
            .group_by("veiculo_id")
            .agg([
                pl.col("km_dia_clean").fill_null(0).max().alias("max_km"),
                pl.col("h_dia_clean").fill_null(0).max().alias("max_h"),
            ])
        )
        # Totalmente vazios: ambos os targets a zero
        empty_vehicles = df_max.filter(
            (pl.col("max_km") == 0) & (pl.col("max_h") == 0)
        )
        empty_ids = set(empty_vehicles["veiculo_id"].to_list())

        # Single-target: apenas um dos targets com dados
        km_only = df_max.filter(
            (pl.col("max_km") > 0) & (pl.col("max_h") == 0)
        )
        km_only_ids = set(km_only["veiculo_id"].to_list())

        h_only = df_max.filter(
            (pl.col("max_km") == 0) & (pl.col("max_h") > 0)
        )
        h_only_ids = set(h_only["veiculo_id"].to_list())

        if verbose:
            if empty_ids:
                print(f"🗑️  Séries vazias (km=0 e h=0): {len(empty_ids):,} veículos removidos")
            if km_only_ids:
                print(f"⚡ Single-target KM (h=0): {len(km_only_ids):,} veículos")
            if h_only_ids:
                print(f"⚡ Single-target H (km=0): {len(h_only_ids):,} veículos")

        df_filtered = df_daily.filter(~pl.col("veiculo_id").is_in(empty_ids))
        return df_filtered, empty_ids, km_only_ids, h_only_ids

    def _get_feature_name(self, extractor: BaseFeatureExtractor, canonical_name: str) -> Optional[str]:
        if not extractor:
            return None
        expanded = extractor.get_expanded_features(canonical_name)
        if isinstance(expanded, list):
            return expanded[0] if expanded else None
        return expanded

    def _compute_thresholds(self, df: pd.DataFrame, verbose: bool = True):
        """Calcula limiares a partir de um DataFrame de features."""
        if verbose:
            print("=" * 80)
            print("CALCULANDO LIMIARES DE QUALIDADE")
            print("=" * 80)
            print()

        thresholds = {}

        # Type features
        prop_km = self._get_feature_name(self.type_extractor, 'proporcao_km')
        corr_km_h = self._get_feature_name(self.type_extractor, 'corr_km_h')

        if prop_km and prop_km in df.columns:
            thresholds["prop_km_high"] = df[prop_km].quantile(self.quantiles['prop_km_high'])
            thresholds["prop_h_low"] = (1 - df[prop_km]).quantile(self.quantiles['prop_h_low'])
            if verbose:
                print(f"✓ {prop_km} alta: > {thresholds['prop_km_high']:.3f}")
                print(f"✓ proporcao_h baixa: < {thresholds['prop_h_low']:.3f}")

        if corr_km_h and corr_km_h in df.columns:
            thresholds["corr_low"] = df[corr_km_h].quantile(self.quantiles['corr_low'])
            if verbose:
                print(f"✓ {corr_km_h} baixa: < {thresholds['corr_low']:.3f}")

        # Segmentation KM features
        taxa_dias_km = self._get_feature_name(self.km_extractor, 'taxa_dias_ativos')
        gap_medio_km = self._get_feature_name(self.km_extractor, 'gap_medio')
        cv_gaps_km = self._get_feature_name(self.km_extractor, 'cv_gaps')

        if taxa_dias_km and taxa_dias_km in df.columns:
            thresholds["taxa_dias_low_km"] = df[taxa_dias_km].quantile(self.quantiles['taxa_dias_low'])
            if verbose:
                print(f"✓ {taxa_dias_km} baixa: < {thresholds['taxa_dias_low_km']:.3f}")

        if gap_medio_km and gap_medio_km in df.columns:
            thresholds["gap_medio_high_km"] = df[gap_medio_km].quantile(self.quantiles['gap_medio_high'])
            if verbose:
                print(f"✓ {gap_medio_km} alto: > {thresholds['gap_medio_high_km']:.1f}")

        if cv_gaps_km and cv_gaps_km in df.columns:
            thresholds["cv_gaps_high_km"] = df[cv_gaps_km].quantile(self.quantiles['cv_gaps_high'])
            if verbose:
                print(f"✓ {cv_gaps_km} alto: > {thresholds['cv_gaps_high_km']:.3f}")

        # Segmentation H features
        taxa_dias_h = self._get_feature_name(self.h_extractor, 'taxa_dias_ativos')
        gap_medio_h = self._get_feature_name(self.h_extractor, 'gap_medio')
        cv_gaps_h = self._get_feature_name(self.h_extractor, 'cv_gaps')

        if taxa_dias_h and taxa_dias_h in df.columns:
            thresholds["taxa_dias_low_h"] = df[taxa_dias_h].quantile(self.quantiles['taxa_dias_low'])
            if verbose:
                print(f"✓ {taxa_dias_h} baixa: < {thresholds['taxa_dias_low_h']:.3f}")

        if gap_medio_h and gap_medio_h in df.columns:
            thresholds["gap_medio_high_h"] = df[gap_medio_h].quantile(self.quantiles['gap_medio_high'])
            if verbose:
                print(f"✓ {gap_medio_h} alto: > {thresholds['gap_medio_high_h']:.1f}")

        if cv_gaps_h and cv_gaps_h in df.columns:
            thresholds["cv_gaps_high_h"] = df[cv_gaps_h].quantile(self.quantiles['cv_gaps_high'])
            if verbose:
                print(f"✓ {cv_gaps_h} alto: > {thresholds['cv_gaps_high_h']:.3f}")

        self.computed_thresholds = thresholds

        if verbose:
            print()
            print("=" * 80)

    def _evaluate_single(self, features: pd.Series, is_single_target: bool = False) -> dict:
        """Avalia qualidade de um único veículo.

        Parameters
        ----------
        features : pd.Series
            Features do veículo.
        is_single_target : bool
            Se True, ignora regras que cruzam km e h (regras 1 e 2)
            e só avalia o target disponível na regra 3.
        """
        quality = SeriesQuality.VALID
        reason = None

        prop_km = self._get_feature_name(self.type_extractor, 'proporcao_km')
        corr_km_h = self._get_feature_name(self.type_extractor, 'corr_km_h')

        # Regras 1 e 2 só fazem sentido para veículos com ambos os targets
        if not is_single_target:
            # Regra 1: Inconsistência física
            if (prop_km and prop_km in features.index
                    and self.computed_thresholds.get("prop_km_high") is not None
                    and self.computed_thresholds.get("prop_h_low") is not None):
                prop_km_val = features[prop_km]
                prop_h_val = 1 - prop_km_val
                if prop_km_val > self.computed_thresholds["prop_km_high"] and prop_h_val < self.computed_thresholds["prop_h_low"]:
                    quality = SeriesQuality.OUTLIER
                    reason = "Inconsistência física: KM sem H"

            # Regra 2: Tipo contraditório
            if (quality == SeriesQuality.VALID and prop_km and corr_km_h
                    and prop_km in features.index and corr_km_h in features.index
                    and self.computed_thresholds.get("prop_km_high") is not None
                    and self.computed_thresholds.get("corr_low") is not None):
                if features[prop_km] > self.computed_thresholds["prop_km_high"] and features[corr_km_h] < self.computed_thresholds["corr_low"]:
                    quality = SeriesQuality.OUTLIER
                    reason = "Tipo contraditório: KM sem correlação com H"

        # Regra 3: Série não-modelável — apenas para o(s) target(s) disponível(is)
        if quality == SeriesQuality.VALID:
            # Determinar quais targets avaliar
            has_km = True
            has_h = True
            if is_single_target:
                # Verificar via proporcao_km: ≈1.0 → km-only, ≈0.0 → h-only
                if prop_km and prop_km in features.index:
                    has_km = features[prop_km] > 0.5
                    has_h = not has_km

            checks = []
            if has_km:
                checks.extend([
                    (self.km_extractor, 'taxa_dias_ativos', 'taxa_dias_low_km', '<'),
                    (self.km_extractor, 'gap_medio', 'gap_medio_high_km', '>'),
                    (self.km_extractor, 'cv_gaps', 'cv_gaps_high_km', '>'),
                ])
            if has_h:
                checks.extend([
                    (self.h_extractor, 'taxa_dias_ativos', 'taxa_dias_low_h', '<'),
                    (self.h_extractor, 'gap_medio', 'gap_medio_high_h', '>'),
                    (self.h_extractor, 'cv_gaps', 'cv_gaps_high_h', '>'),
                ])
            not_modelable = False
            for extractor, canonical, threshold_attr, op in checks:
                feat_name = self._get_feature_name(extractor, canonical)
                threshold = self.computed_thresholds.get(threshold_attr)
                if feat_name and feat_name in features.index and threshold is not None:
                    val = features[feat_name]
                    if (op == '<' and val < threshold) or (op == '>' and val > threshold):
                        not_modelable = True

            if not_modelable:
                quality = SeriesQuality.NOT_MODELABLE
                reason = "Série não-modelável"

        # Veículos single-target válidos recebem qualidade SINGLE_TARGET
        if is_single_target and quality == SeriesQuality.VALID:
            quality = SeriesQuality.SINGLE_TARGET
            reason = "Single-target: apenas um target disponível"

        return {
            "quality": quality,
            "quality_reason": reason,
        }

    def _evaluate_batch(self, df: pd.DataFrame, single_target_ids: set = None, verbose: bool = True) -> pd.DataFrame:
        """Avalia qualidade de múltiplas amostras em lote."""
        if single_target_ids is None:
            single_target_ids = set()

        if verbose:
            print("=" * 80)
            print("AVALIANDO QUALIDADE DOS DADOS")
            print("=" * 80)
            print()

        results = [
            self._evaluate_single(
                row,
                is_single_target=(row["veiculo_id"] in single_target_ids),
            )
            for _, row in df.iterrows()
        ]
        df_result = df[['veiculo_id']].copy()
        df_result['quality'] = [r['quality'] for r in results]
        df_result['quality_reason'] = [r['quality_reason'] for r in results]

        if verbose:
            total = len(df_result)
            n_outliers = (df_result['quality'] == SeriesQuality.OUTLIER).sum()
            n_not_modelable = (df_result['quality'] == SeriesQuality.NOT_MODELABLE).sum()
            n_single_target = (df_result['quality'] == SeriesQuality.SINGLE_TARGET).sum()
            n_valid = (df_result['quality'] == SeriesQuality.VALID).sum()
            print(f"Total de amostras:        {total:>8,}")
            print(f"Outliers físicos:         {n_outliers:>8,} ({n_outliers/total*100:5.1f}%)")
            print(f"Séries não-modeláveis:    {n_not_modelable:>8,} ({n_not_modelable/total*100:5.1f}%)")
            print(f"Single-target:            {n_single_target:>8,} ({n_single_target/total*100:5.1f}%)")
            print(f"Amostras válidas:         {n_valid:>8,} ({n_valid/total*100:5.1f}%)")
            print()
            print("=" * 80)

        return df_result

    def _save_artifacts(
        self,
        df: pd.DataFrame,
        output_dir: Path,
        all_ids,
        formatted_ids,
        failed_ids,
    ):
        """Salva CSVs e JSON de anomalias detectadas."""
        anomalies_dir = output_dir / "anomalies"
        anomalies_dir.mkdir(parents=True, exist_ok=True)

        if (df["quality"] == SeriesQuality.OUTLIER).sum() > 0:
            df[df["quality"] == SeriesQuality.OUTLIER].to_csv(anomalies_dir / "outliers_fisicos.csv", index=False)
            print(f"✓ Outliers físicos: {anomalies_dir / 'outliers_fisicos.csv'}")

        if (df["quality"] == SeriesQuality.NOT_MODELABLE).sum() > 0:
            df[df["quality"] == SeriesQuality.NOT_MODELABLE].to_csv(
                anomalies_dir / "series_nao_modelaveis.csv", index=False,
            )
            print(f"✓ Séries não-modeláveis: {anomalies_dir / 'series_nao_modelaveis.csv'}")

        type_features = self.type_extractor.selected_features if self.type_extractor else None
        km_features = self.km_extractor.selected_features if self.km_extractor else None
        h_features = self.h_extractor.selected_features if self.h_extractor else None

        summary = {
            "total_veiculos": int(len(df)),
            "outliers_fisicos": int((df["quality"] == SeriesQuality.OUTLIER).sum()),
            "series_nao_modelaveis": int((df["quality"] == SeriesQuality.NOT_MODELABLE).sum()),
            "series_vazias": int((df["quality"] == SeriesQuality.EMPTY).sum()),
            "single_target": int((df["quality"] == SeriesQuality.SINGLE_TARGET).sum()),
            "veiculos_validos": int((df["quality"] == SeriesQuality.VALID).sum()),
            "filter_thresholds": self.filter_thresholds,
            "computed_thresholds": self.computed_thresholds,
            "quantiles": self.quantiles,
            "features_usadas": {
                "type": type_features,
                "km": km_features,
                "h": h_features,
            },
            "formatacao": {
                "veiculos_totais": len(all_ids),
                "veiculos_formatados": len(formatted_ids),
                "veiculos_falharam": len(failed_ids),
            },
        }
        with open(anomalies_dir / "detection_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Resumo: {anomalies_dir / 'detection_summary.json'}")

        config = {
            'computed_thresholds': self.computed_thresholds,
            'quantiles': self.quantiles,
            'features': {
                'type': type_features,
                'km': km_features,
                'h': h_features,
            },
        }
        config_path = anomalies_dir / "quality_configs.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Limiares salvos em: {config_path}")
        print()

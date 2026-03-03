import polars as pl
import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from typing import List, Union, Literal

from darts import TimeSeries
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# FEATURE EXTRACTORS
# ============================================================================

class BaseFeatureExtractor(ABC):
    """Classe base para extração de features"""
    
    def __init__(self, name: str, feature_names: List[str], required_targets: List[str]):
        self.name = name
        self.feature_names = feature_names
        self.required_targets = required_targets
    
    @abstractmethod
    def extract(self, df: Union[pl.DataFrame, pd.DataFrame]) -> pd.DataFrame:
        """Extrai features do DataFrame"""
        pass
    
    def extract_from_timeseries(self, ts: TimeSeries, veiculo_id: int) -> pd.DataFrame:
        """Extrai features de uma TimeSeries do Darts"""
        df_ts = ts.pd_dataframe().reset_index()
        
        if len(df_ts.columns) == 2:
            df_ts.columns = ['data', self.required_targets[0]]
        elif len(df_ts.columns) == 3:
            df_ts.columns = ['data', 'km_dia_clean', 'h_dia_clean']
        else:
            raise ValueError(f"TimeSeries deve ter 1 ou 2 componentes")
        
        df_ts['veiculo_id'] = veiculo_id
        return self.extract(df_ts)
    
    def _validate_targets(self, df: pl.DataFrame):
        """Valida targets"""
        missing = [t for t in self.required_targets if t not in df.columns]
        if missing:
            raise ValueError(f"Targets faltando: {missing}")
    
    def _to_polars(self, df: Union[pl.DataFrame, pd.DataFrame]) -> pl.DataFrame:
        """Converte para Polars"""
        if isinstance(df, pd.DataFrame):
            df_pl = pl.from_pandas(df)
        else:
            df_pl = df.clone()
        
        if df_pl['data'].dtype not in [pl.Date, pl.Datetime]:
            df_pl = df_pl.with_columns(pl.col('data').str.strptime(pl.Date, '%Y-%m-%d'))
        
        return df_pl
    
    def _get_effective_period(self, df: pl.DataFrame, target: str, metric: str) -> pl.DataFrame:
        """Calcula período efetivo para um target específico"""
        return (
            df.filter(pl.col(target) > 0)
            .group_by(['veiculo_id', 'ano', 'semana'])
            .agg(pl.col('data').min().alias('primeira_data_semana')) 
            .group_by('veiculo_id')
            .agg([
                pl.col('primeira_data_semana').min().alias('primeira'), 
                pl.col('primeira_data_semana').max().alias('ultima')
            ])
            .with_columns([
                (pl.col('primeira') - pl.duration(days=pl.col('primeira').dt.weekday())).alias(f'inicio_periodo_{metric}'),
                (pl.col('ultima') + pl.duration(days=6 - pl.col('ultima').dt.weekday())).alias(f'fim_periodo_{metric}')
            ])
            .select(['veiculo_id', f'inicio_periodo_{metric}', f'fim_periodo_{metric}'])
        )


class TypeFeatureExtractor(BaseFeatureExtractor):
    """
    Extrator de Features para Classificação de Tipo (Etapa 1)
    
    Input obrigatório: veiculo_id, data, km_dia_clean, h_dia_clean
    
    Extrai APENAS 4 features de comparação entre km e h
    """
    
    def __init__(self):
        features = [
            'corr_km_h',
            'razao_km_h',
            'proporcao_km',
            'proporcao_h'
        ]
        super().__init__('type_features', features, 
                        required_targets=['km_dia_clean', 'h_dia_clean'])
    
    def extract(self, df: Union[pl.DataFrame, pd.DataFrame]) -> pd.DataFrame:
        """Extrai 4 features de tipo"""
        df_pl = self._to_polars(df)
        self._validate_targets(df_pl)
        
        df_pl = df_pl.with_columns([
            pl.col('data').dt.year().alias('ano'),
            pl.col('data').dt.week().alias('semana')
        ])
        
        # Períodos efetivos
        periodo_km = self._get_effective_period(df_pl, 'km_dia_clean', 'km')
        periodo_h = self._get_effective_period(df_pl, 'h_dia_clean', 'h')
        periodos = periodo_km.join(periodo_h, on='veiculo_id', how='full')
        
        df_pl = df_pl.join(periodos, on='veiculo_id', how='left')
        
        df_pl = df_pl.with_columns([
            ((pl.col('data') >= pl.col('inicio_periodo_km')) & 
             (pl.col('data') <= pl.col('fim_periodo_km'))).alias('dentro_km'),
            ((pl.col('data') >= pl.col('inicio_periodo_h')) & 
             (pl.col('data') <= pl.col('fim_periodo_h'))).alias('dentro_h')
        ])
        
        # Agregar (apenas total e dias para calcular {metric}_por_dia)
        agg_km = (
            df_pl.filter(pl.col('dentro_km'))
            .group_by('veiculo_id')
            .agg([
                pl.col('km_dia_clean').sum().alias('total_km'),
                pl.col('data').n_unique().alias('dias_km')
            ])
        )
        
        agg_h = (
            df_pl.filter(pl.col('dentro_h'))
            .group_by('veiculo_id')
            .agg([
                pl.col('h_dia_clean').sum().alias('total_h'),
                pl.col('data').n_unique().alias('dias_h')
            ])
        )
        
        # Correlação
        corr = (
            df_pl.filter(
                (pl.col('dentro_km') | pl.col('dentro_h')) &
                ((pl.col('km_dia_clean') > 0) | (pl.col('h_dia_clean') > 0))
            )
            .group_by('veiculo_id')
            .agg(pl.corr('km_dia_clean', 'h_dia_clean').alias('corr_km_h'))
        )
        
        # Combinar
        features = (
            agg_km
            .join(agg_h, on='veiculo_id', how='full')
            .join(corr, on='veiculo_id', how='left')
            .fill_null(0)
        )
        
        df_final = features.to_pandas()
        
        # Calcular features temporárias para derivar as finais
        km_por_dia = np.where(df_final['dias_km'] > 0, df_final['total_km'] / df_final['dias_km'], 0)
        horas_por_dia = np.where(df_final['dias_h'] > 0, df_final['total_h'] / df_final['dias_h'], 0)
        
        # Calcular as 4 features finais
        df_final['razao_km_h'] = np.where(horas_por_dia > 0, km_por_dia / horas_por_dia, 999999)
        df_final['proporcao_km'] = km_por_dia / (km_por_dia + horas_por_dia + 0.001)
        df_final['proporcao_h'] = horas_por_dia / (km_por_dia + horas_por_dia + 0.001)
        
        df_final = df_final.replace([np.inf, -np.inf], 999999).fillna(0)
        
        return df_final[['veiculo_id'] + self.feature_names]


class SegmentationFeatureExtractor(BaseFeatureExtractor):
    """
    Extrator de Features para Segmentação (Etapa 2)
    
    Trabalha com APENAS UM target
    Extrai 15 features
    """
    
    def __init__(self, metric: Literal['km', 'h']):
        self.metric = metric
        target_col = f'{metric}_dia_clean'
        
        features = [
            f'{metric}_por_dia',
            f'media_{metric}',
            f'max_{metric}',
            f'mediana_{metric}',
            f'std_{metric}',
            f'score_continuidade_{metric}',
            f'taxa_semanas_ativas_{metric}',
            f'taxa_dias_ativos_{metric}',
            f'cv_gaps_{metric}',
            f'gap_medio_{metric}',
            f'gap_max_{metric}',
            f'cv_{metric}',
            f'p25_{metric}',
            f'p75_{metric}',
            f'iqr_{metric}'
        ]
        
        name = f'segmentation_features_{metric}'
        super().__init__(name, features, required_targets=[target_col])
    
    def extract(self, df: Union[pl.DataFrame, pd.DataFrame]) -> pd.DataFrame:
        """Extrai 15 features de segmentação"""
        df_pl = self._to_polars(df)
        self._validate_targets(df_pl)
        
        target = f'{self.metric}_dia_clean'
        
        df_pl = df_pl.with_columns([
            pl.col('data').dt.year().alias('ano'),
            pl.col('data').dt.week().alias('semana')
        ])
        
        # Período
        periodo = self._get_effective_period(df_pl, target, self.metric)
        df_pl = df_pl.join(periodo, on='veiculo_id', how='left')
        
        df_pl = df_pl.with_columns([
            ((pl.col('data') >= pl.col(f'inicio_periodo_{self.metric}')) & 
             (pl.col('data') <= pl.col(f'fim_periodo_{self.metric}'))).alias('dentro')
        ])
        
        # Agregar
        agg = (
            df_pl.filter(pl.col('dentro'))
            .group_by('veiculo_id')
            .agg([
                pl.col(target).sum().alias('total'),
                pl.col(target).mean().alias('media'),
                pl.col(target).std().alias('std'),
                pl.col(target).median().alias('mediana'),
                pl.col(target).max().alias('max'),
                pl.col(target).quantile(0.25).alias('p25'),
                pl.col(target).quantile(0.75).alias('p75'),
                (pl.col(target) > 0).sum().alias('dias_ativos'),
                pl.col('data').n_unique().alias('total_dias'),
                pl.struct(['ano', 'semana']).n_unique().alias('total_semanas'),
                pl.struct(['ano', 'semana']).filter(pl.col(target) > 0).n_unique().alias('semanas_ativas')
            ])
        )
        
        # Gaps
        gaps = (
            df_pl.filter(pl.col('dentro') & (pl.col(target) > 0))
            .sort(['veiculo_id', 'data'])
            .with_columns(pl.col('data').diff().over('veiculo_id').dt.total_days().alias('gap'))
            .filter(pl.col('gap').is_not_null())
            .group_by('veiculo_id')
            .agg([
                pl.col('gap').mean().alias('gap_medio'),
                pl.col('gap').max().alias('gap_max'),
                pl.col('gap').std().alias('gap_std')
            ])
        )
        
        # Combinar
        features = agg.join(gaps, on='veiculo_id', how='left').fill_null(0)
        df_final = features.to_pandas()
        
        # Calcular as 15 features finais 
        suffix = self.metric
        
        df_final[f'{suffix}_por_dia'] = np.where(df_final['total_dias'] > 0, df_final['total'] / df_final['total_dias'], 0)
        df_final[f'media_{suffix}'] = df_final['media']
        df_final[f'max_{suffix}'] = df_final['max']
        df_final[f'mediana_{suffix}'] = df_final['mediana']
        df_final[f'std_{suffix}'] = df_final['std']
        df_final[f'p25_{suffix}'] = df_final['p25']
        df_final[f'p75_{suffix}'] = df_final['p75']
        df_final[f'gap_medio_{suffix}'] = df_final['gap_medio']
        df_final[f'gap_max_{suffix}'] = df_final['gap_max']
        
        df_final[f'taxa_dias_ativos_{suffix}'] = np.where(df_final['total_dias'] > 0, df_final['dias_ativos'] / df_final['total_dias'], 0)
        df_final[f'taxa_semanas_ativas_{suffix}'] = np.where(df_final['total_semanas'] > 0, df_final['semanas_ativas'] / df_final['total_semanas'], 0)
        df_final[f'cv_{suffix}'] = np.where(df_final['media'] > 0, df_final['std'] / df_final['media'], 0)
        df_final[f'cv_gaps_{suffix}'] = np.where(df_final['gap_medio'] > 0, df_final['gap_std'] / df_final['gap_medio'], 0)
        df_final[f'iqr_{suffix}'] = df_final[f'p75_{suffix}'] - df_final[f'p25_{suffix}']
        df_final[f'score_continuidade_{suffix}'] = df_final[f'taxa_semanas_ativas_{suffix}'] * (1 / (1 + df_final[f'cv_gaps_{suffix}']))
                
        df_final = df_final.replace([np.inf, -np.inf], 999999).fillna(0)
        
        return df_final[['veiculo_id'] + self.feature_names]
    

class WeekdayFeatureExtractor(BaseFeatureExtractor):
    """
    Extrator de Features por Dia da Semana (70 métricas)
    
    Trabalha com APENAS UM target.
    Extrai 10 features para cada dia da semana (1=seg, 7=dom).
    """
    
    def __init__(self, metric: Literal['km', 'h']):
        self.metric = metric
        target_col = f'{metric}_dia_clean'
        
        # 7 dias × 10 features = 70 features
        features = []
        for day in range(1, 8):
            for feat in ['mean', 'std', 'median', 'max', 'min', 'p25', 'p75', 'iqr', 'prob_active', 'cv']:
                features.append(f'day_{day}_{feat}')
        
        name = f'weekday_features_{metric}'
        super().__init__(name, features, required_targets=[target_col])
    
    def extract(self, df: pd.DataFrame | pl.DataFrame) -> pd.DataFrame:
        """
        Extrai 70 features de dia da semana
        
        Parameters
        ----------
        df : pd.DataFrame ou pl.DataFrame
            DataFrame com colunas: veiculo_id, data, {metric}_dia_clean
        
        Returns
        -------
        pd.DataFrame
            DataFrame com veiculo_id e 70 features (7 dias × 10 features)
        """
        df_pl = self._to_polars(df)
        self._validate_targets(df_pl)
        
        target = f'{self.metric}_dia_clean'
        
        # Adicionar chaves temporais usadas no período efetivo + dia da semana.
        df_pl = df_pl.with_columns([
            pl.col('data').dt.year().alias('ano'),
            pl.col('data').dt.week().alias('semana'),
            pl.col('data').dt.weekday().alias('weekday')  # 0=seg, 6=dom
        ]).with_columns([
            (pl.col('weekday') + 1).alias('day')  # 1=seg, 7=dom
        ])
        
        # Calcular período efetivo
        periodo = self._get_effective_period(df_pl, target, self.metric)
        df_pl = df_pl.join(periodo, on='veiculo_id', how='left')
        
        # Filtrar período efetivo
        df_pl = df_pl.filter(
            (pl.col('data') >= pl.col(f'inicio_periodo_{self.metric}')) &
            (pl.col('data') <= pl.col(f'fim_periodo_{self.metric}'))
        )
        
        # Agregar por veículo e dia da semana
        agg = (
            df_pl.group_by(['veiculo_id', 'day'])
            .agg([
                pl.col(target).mean().alias('mean'),
                pl.col(target).std().alias('std'),
                pl.col(target).median().alias('median'),
                pl.col(target).max().alias('max'),
                pl.col(target).min().alias('min'),
                pl.col(target).quantile(0.25).alias('p25'),
                pl.col(target).quantile(0.75).alias('p75'),
                (pl.col(target) > 0).mean().alias('prob_active'),
                pl.col('data').n_unique().alias('n_occurrences')
            ])
        )
        
        df_agg = agg.to_pandas()
        
        # Calcular features derivadas
        df_agg['iqr'] = df_agg['p75'] - df_agg['p25']
        df_agg['cv'] = np.where(df_agg['mean'] > 0, df_agg['std'] / df_agg['mean'], 0)
        
        # Limpar infinitos e NaN
        df_agg = df_agg.replace([np.inf, -np.inf], 999999).fillna(0)
        
        # Pivotar para formato wide (uma linha por veículo)
        df_pivot = df_agg.pivot_table(
            index='veiculo_id',
            columns='day',
            values=['mean', 'std', 'median', 'max', 'min', 'p25', 'p75', 'iqr', 'prob_active', 'cv']
        )
        
        # Renomear colunas para day_X_feature
        df_pivot.columns = [f'day_{day}_{feat}' for feat, day in df_pivot.columns]
        df_pivot = df_pivot.reset_index()
        
        # Garantir que todas as 70 features existem (preencher com 0 se faltando)
        for feat_name in self.feature_names:
            if feat_name not in df_pivot.columns:
                df_pivot[feat_name] = 0.0
        
        return df_pivot[['veiculo_id'] + self.feature_names]    

import polars as pl
import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from typing import Dict, List, Union, Literal, Optional

from moviasai.profiling.utils import compute_effective_period

import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# BASE FEATURE EXTRACTOR
# ============================================================================

class BaseFeatureExtractor(ABC):
    """Classe base para extração de features com período efetivo centralizado"""
    
    # Prefixos padrão para cada tipo de extractor
    PREFIX = None  # Deve ser definido nas subclasses
    NORM_RULES: Dict[str, str] = {}
    
    def __init__(self, name: str, feature_names: List[str], required_targets: List[str]):
        """
        Parameters
        ----------
        name : str
            Nome identificador do extractor
        feature_names : List[str]
            Lista completa de features disponíveis (após expansão, sem sufixo)
        required_targets : List[str]
            Targets necessários (ex: ['km_dia_clean'])
        """
        self.name = name
        self.all_feature_names = feature_names
        self._feature_names = feature_names.copy()
        self.required_targets = required_targets
        self._feature_map = self.feature_map
    
    @property
    def feature_names(self) -> List[str]:
        """Retorna lista de nomes expandidos das features"""
        return self._feature_names
    
    @property
    def selected_features(self) -> List[str]:
        """Features canônicas selecionadas pelo usuário"""
        return self._feature_names
    # Adicionar na classe BaseFeatureExtractor

    @property
    def feature_map(self) -> Dict[str, Union[str, List[str]]]:
        """
        Retorna mapeamento de features canônicas para features expandidas
        
        Returns
        -------
        Dict[str, Union[str, List[str]]]
            Dicionário onde:
            - chave: nome da feature canônica (de selected_features)
            - valor: string (1:1) ou lista (1:N) com nomes expandidos (de feature_names)
        
        Examples
        --------
        SegmentationFeatureExtractor:
            {'media': 'seg_media_km', 'cv': 'seg_cv_km'}  # 1:1
        
        WeekdayFeatureExtractor:
            {'mean': ['day_1_mean_km', 'day_2_mean_km', ..., 'day_7_mean_km']}  # 1:N
        
        MonthPhaseFeatureExtractor:
            {'mean': ['phase_mean_inicio_km', 'phase_mean_meio_km', 'phase_mean_fim_km']}
        """
        # Método padrão (deve ser sobrescrito nas subclasses se necessário)
        return {feat: feat for feat in self.selected_features}
    
    def get_expanded_features(self, base_feature: str) -> Union[str, List[str]]:
        if base_feature in self.selected_features:
            return self._feature_map[base_feature]

    def get_canonical_features(self, expanded_features: List[str]) -> List[str]:
        """Devolve a forma canónica de cada feature expandida.

        Parameters
        ----------
        expanded_features : List[str]
            Nomes expandidos (e.g. ``['seg_media_km', 'day_1_mean_km']``).

        Returns
        -------
        List[str]
            Lista (mesma ordem, sem duplicados consecutivos) com nomes
            canónicos correspondentes.  Features não reconhecidas são
            devolvidas tal como estão.
        """
        # Construir mapa inverso: expanded -> canonical
        inv: Dict[str, str] = {}
        for canonical, expanded in self._feature_map.items():
            if isinstance(expanded, list):
                for e in expanded:
                    inv[e] = canonical
            else:
                inv[expanded] = canonical

        seen: set = set()
        result: List[str] = []
        for feat in expanded_features:
            canonical = inv.get(feat, feat)
            if canonical not in seen:
                seen.add(canonical)
                result.append(canonical)
        return result

    def get_canonical_map(self, expanded_features: List[str]) -> Dict[str, List[str]]:
        """Agrupa features expandidas pela sua forma canónica.

        Parameters
        ----------
        expanded_features : List[str]
            Nomes expandidos (e.g. ``['day_1_mean_km', 'day_2_mean_km']``).

        Returns
        -------
        Dict[str, List[str]]
            Mapeamento canónica → lista de expandidas (sem duplicatas),
            na ordem de aparição.
        """
        inv: Dict[str, str] = {}
        for canonical, expanded in self._feature_map.items():
            if isinstance(expanded, list):
                for e in expanded:
                    inv[e] = canonical
            else:
                inv[expanded] = canonical

        result: Dict[str, List[str]] = {}
        seen: set = set()
        for feat in expanded_features:
            if feat in seen:
                continue
            seen.add(feat)
            canonical = inv.get(feat, feat)
            result.setdefault(canonical, []).append(feat)
        return result

    @abstractmethod
    def _extract_features(self, df_pl: pl.DataFrame) -> pd.DataFrame:
        """
        Método abstrato para extração de features (DEVE ser implementado)
        
        Recebe DataFrame Polars já:
        - Validado (targets existem)
        - Com colunas auxiliares (ano, semana, weekday se necessário)
        - Com período efetivo aplicado (coluna 'dentro' ou 'dentro_{metric}')
        
        Parameters
        ----------
        df_pl : pl.DataFrame
            DataFrame preparado
        
        Returns
        -------
        pd.DataFrame
            DataFrame com veiculo_id e features extraídas
        """
        pass
    
    def extract(self, df: Union[pl.DataFrame, pd.DataFrame]) -> pd.DataFrame:
        """
        Extrai features do DataFrame (MÉTODO FINAL - NÃO SOBRESCREVER)
        
        Este método:
        1. Converte para Polars
        2. Valida targets
        3. Adiciona colunas auxiliares (ano, semana, weekday)
        4. Calcula e aplica período efetivo
        5. Chama _extract_features() implementado pela subclasse
        """
        df_pl = self._to_polars(df)
        self._validate_targets(df_pl)
        df_pl = self._prepare_dataframe(df_pl)
        df_pl = self._apply_effective_period(df_pl)
        return self._extract_features(df_pl)
    
    def _prepare_dataframe(self, df_pl: pl.DataFrame) -> pl.DataFrame:
        """Prepara DataFrame adicionando colunas auxiliares necessárias"""
        columns_to_add = [
            pl.col('data').dt.year().alias('ano'),
            pl.col('data').dt.week().alias('semana')
        ]
        
        if self._needs_weekday():
            columns_to_add.append(
                pl.col('data').dt.weekday().alias('weekday')
            )
        
        return df_pl.with_columns(columns_to_add)
    
    def _needs_weekday(self) -> bool:
        """Indica se a subclasse precisa da coluna weekday (sobrescrever se necessário)"""
        return False
    
    def _apply_effective_period(self, df_pl: pl.DataFrame) -> pl.DataFrame:
        """
        Aplica período efetivo ao DataFrame
        
        Para cada target:
        1. Calcula período efetivo
        2. Faz join
        3. Cria coluna 'dentro_{metric}'
        """
        if not self.required_targets:
            return df_pl
        
        for target in self.required_targets:
            metric = target.split('_')[0]
            
            periodo = compute_effective_period(df_pl, target).rename({
                'dt_inicio': f'inicio_periodo_{metric}',
                'dt_fim': f'fim_periodo_{metric}',
            })
            df_pl = df_pl.join(periodo, on='veiculo_id', how='left')
            
            df_pl = df_pl.with_columns([
                (
                    (pl.col('data') >= pl.col(f'inicio_periodo_{metric}')) &
                    (pl.col('data') <= pl.col(f'fim_periodo_{metric}'))
                ).alias(f'dentro_{metric}')
            ])
        
        if len(self.required_targets) == 1:
            metric = self.required_targets[0].split('_')[0]
            df_pl = df_pl.with_columns(
                pl.col(f'dentro_{metric}').alias('dentro')
            )
        
        return df_pl
    
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


# ============================================================================
# TYPE FEATURE EXTRACTOR
# ============================================================================

class TypeFeatureExtractor(BaseFeatureExtractor):
    """
    Extrator de Features para Classificação de Tipo (Etapa 1)
    Prefixo: Nenhum (features únicas, sem necessidade)
    """
    
    PREFIX = None  # Sem prefixo
    
    def __init__(self, features: Optional[List[str]] = None):
        """
        Parameters
        ----------
        features : Optional[List[str]]
            Features base a extrair: ['corr_km_h', 'razao_km_h', 'proporcao_km']
            None = todas
        """
        self._base_features = ['corr_km_h', 'razao_km_h', 'proporcao_km']
        self._selected_base = features if features is not None else self._base_features.copy()
        
        invalid = [f for f in self._selected_base if f not in self._base_features]
        if invalid:
            raise ValueError(f"Features inválidas: {invalid}")
        
        self._expanded_features = self._selected_base.copy()
        self._final_features = self._selected_base.copy()
        
        super().__init__('type_features', self._base_features, 
                        required_targets=['km_dia_clean', 'h_dia_clean'])
        
        self._feature_names = self._final_features
    
    @property
    def selected_features(self) -> List[str]:
        """Features canônicas selecionadas pelo usuário"""
        return self._selected_base.copy()
    
    @property
    def feature_names(self) -> List[str]:
        """Nomes finais das features"""
        return self._final_features.copy()
    
    def _extract_features(self, df_pl: pl.DataFrame) -> pd.DataFrame:
        """Extrai features de tipo"""
        all_vehicles = df_pl.select('veiculo_id').unique()
        
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
        
        corr = None
        if 'corr_km_h' in self._selected_base:
            corr = (
                df_pl.filter(
                    pl.col('dentro_km') &
                    pl.col('dentro_h') &
                    (pl.col('km_dia_clean') > 0) &
                    (pl.col('h_dia_clean') > 0)
                )
                .group_by('veiculo_id')
                .agg([
                    pl.count().alias('n_corr'),
                    pl.corr('km_dia_clean', 'h_dia_clean').alias('corr_raw')
                ])
                .with_columns(
                    pl.when(pl.col('n_corr') < 5)
                    .then(0.0)
                    .otherwise(pl.col('corr_raw'))
                    .alias('corr_km_h')
                )
                .select(['veiculo_id', 'corr_km_h'])
            )
        
        features = all_vehicles.join(agg_km, on='veiculo_id', how='left')
        features = features.join(agg_h, on='veiculo_id', how='left')
        
        if corr is not None:
            features = features.join(corr, on='veiculo_id', how='left')
        
        features = features.fill_null(0)
        df_final = features.to_pandas()
        
        km_por_dia = np.where(
            df_final['dias_km'] > 0,
            df_final['total_km'] / df_final['dias_km'], 
            0
        )
        
        h_por_dia = np.where(
            df_final['dias_h'] > 0,
            df_final['total_h'] / df_final['dias_h'], 
            0
        )
        
        result_df = df_final[['veiculo_id']].copy()
        
        if 'corr_km_h' in self._selected_base:
            result_df['corr_km_h'] = df_final['corr_km_h']
        
        if 'razao_km_h' in self._selected_base:
            raw_ratio = np.where(h_por_dia > 0, km_por_dia / h_por_dia, 0)
            log_ratio = np.log1p(raw_ratio)
            result_df['razao_km_h'] = np.clip(log_ratio, 0, 5)
        
        if 'proporcao_km' in self._selected_base:
            denom = km_por_dia + h_por_dia + 1e-6
            result_df['proporcao_km'] = km_por_dia / denom
        
        result_df = result_df.replace([np.inf, -np.inf], 0).fillna(0)
        
        if (result_df['veiculo_id'] == 0).any():
            warnings.warn("AVISO: Detectado veiculo_id = 0")
            result_df = result_df[result_df['veiculo_id'] != 0]
        
        return result_df[['veiculo_id'] + self.feature_names]


# ============================================================================
# SEGMENTATION FEATURE EXTRACTOR
# ============================================================================

class SegmentationFeatureExtractor(BaseFeatureExtractor):
    """
    Extrator de Features para Segmentação (Etapa 2)
    Prefixo: 'seg_'
    """
    
    PREFIX = 'seg'
    NORM_RULES = {
        'media': 'by_upper', 'mediana': 'by_upper', 'std': 'by_upper',
        'p25': 'by_upper', 'p75': 'by_upper', 'iqr': 'by_upper',
        'taxa_dias_ativos': 'no_norm', 'taxa_semanas_ativas': 'no_norm',
        'score_continuidade': 'no_norm',
        'cv': 'log1p', 'cv_gaps': 'log1p',
        'max': 'by_upper', 'gap_max': 'global', 'p95': 'by_upper',
        'tamanho_periodo': 'global', 'gap_medio': 'global',
    }
    
    def __init__(
        self, 
        metric: Literal['km', 'h'], 
        features: Optional[List[str]] = None,
        add_suffix: bool = True
    ):
        """
        Parameters
        ----------
        metric : Literal['km', 'h']
            Métrica a ser analisada
        features : Optional[List[str]]
            Features base a extrair (None = todas)
        add_suffix : bool
            Se True, adiciona sufixo da métrica (padrão: True)
        """
        self.metric = metric
        self.add_suffix = add_suffix
        target_col = f'{metric}_dia_clean'
        
        self._base_features = [
            'media', 'max', 'mediana', 'std', 'score_continuidade',
            'taxa_semanas_ativas', 'taxa_dias_ativos', 'cv_gaps',
            'gap_medio', 'gap_max', 'cv', 'p25', 'p75', 'p95', 'iqr', 'tamanho_periodo'
        ]
        
        self._selected_base = features if features is not None else self._base_features.copy()
        
        invalid = [f for f in self._selected_base if f not in self._base_features]
        if invalid:
            raise ValueError(f"Features inválidas: {invalid}")
        
        # Adicionar prefixo 'seg_'
        self._expanded_features = [f'{self.PREFIX}_{feat}' for feat in self._selected_base]
        
        if self.add_suffix:
            self._final_features = [f'{feat}_{self.metric}' for feat in self._expanded_features]
        else:
            self._final_features = self._expanded_features.copy()
        
        name = f'segmentation_features_{metric}'
        super().__init__(name, [f'{self.PREFIX}_{f}' for f in self._base_features], required_targets=[target_col])
        
        self._feature_names = self._final_features
    
    @property
    def selected_features(self) -> List[str]:
        """Features canônicas selecionadas pelo usuário"""
        return self._selected_base.copy()
    
    @property
    def feature_names(self) -> List[str]:
        """Nomes finais das features (com prefixo + sufixo)"""
        return self._final_features.copy()
    
    @property
    def feature_map(self) -> Dict[str, str]:
        """Mapeamento 1:1 com prefixo + sufixo"""
        result = {}
        for feat_base in self._selected_base:
            expanded = f'{self.PREFIX}_{feat_base}'
            if self.add_suffix:
                expanded = f'{expanded}_{self.metric}'
            result[feat_base] = expanded
        return result

    def _needs_gaps(self) -> bool:
        gap_features = ['gap_medio', 'gap_max', 'cv_gaps', 'score_continuidade']
        return any(f in self._selected_base for f in gap_features)
    
    def _extract_features(self, df_pl: pl.DataFrame) -> pd.DataFrame:
        """Extrai features de segmentação"""
        target = f'{self.metric}_dia_clean'
        base_features = self._selected_base
        
        agg_exprs = [pl.col(target).sum().alias('total')]
        
        if 'tamanho_periodo' in base_features:
            agg_exprs.extend([
                pl.col(f'inicio_periodo_{self.metric}').first().alias('inicio_periodo'),
                pl.col(f'fim_periodo_{self.metric}').first().alias('fim_periodo')
            ])
        
        if any(f in base_features for f in ['media', 'cv']):
            agg_exprs.append(pl.col(target).mean().alias('media'))
        
        if any(f in base_features for f in ['std', 'cv']):
            agg_exprs.append(pl.col(target).std().alias('std'))
        
        if 'mediana' in base_features:
            agg_exprs.append(pl.col(target).median().alias('mediana'))
      
        if 'max' in base_features:
            agg_exprs.append(pl.col(target).max().alias('max'))
        
        if 'p95' in base_features:
            agg_exprs.append(pl.col(target).quantile(0.95).alias('p95'))
        
        if any(f in base_features for f in ['p25', 'iqr']):
            agg_exprs.append(pl.col(target).quantile(0.25).alias('p25'))
        
        if any(f in base_features for f in ['p75', 'iqr']):
            agg_exprs.append(pl.col(target).quantile(0.75).alias('p75'))
        
        if 'taxa_dias_ativos' in base_features:
            agg_exprs.append((pl.col(target) > 0).sum().alias('dias_ativos'))
        
        agg_exprs.extend([
            pl.col('data').n_unique().alias('total_dias'),
            pl.struct(['ano', 'semana']).n_unique().alias('total_semanas')
        ])
        
        if any(f in base_features for f in ['taxa_semanas_ativas', 'score_continuidade']):
            agg_exprs.append(
                pl.struct(['ano', 'semana']).filter(pl.col(target) > 0).n_unique().alias('semanas_ativas')
            )
        
        agg = (
            df_pl.filter(pl.col('dentro'))
            .group_by('veiculo_id')
            .agg(agg_exprs)
        )
        
        if self._needs_gaps():
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
            features = agg.join(gaps, on='veiculo_id', how='left')
        else:
            features = agg
        
        features = features.fill_null(0)
        df_final = features.to_pandas()
        
        result_df = df_final[['veiculo_id']].copy()
        
        # Criar colunas SEM prefixo primeiro, depois renomear
        if 'tamanho_periodo' in base_features:
            result_df['tamanho_periodo'] = (
                pd.to_datetime(df_final['fim_periodo']) - 
                pd.to_datetime(df_final['inicio_periodo'])
            ).dt.days + 1
        
        if 'media' in base_features:
            result_df['media'] = df_final.get('media', 0)
        
        if 'max' in base_features:
            result_df['max'] = df_final.get('max', 0)
        
        if 'mediana' in base_features:
            result_df['mediana'] = df_final.get('mediana', 0)
        
        if 'std' in base_features:
            result_df['std'] = df_final.get('std', 0)
        
        if 'p25' in base_features:
            result_df['p25'] = df_final.get('p25', 0)
        
        if 'p75' in base_features:
            result_df['p75'] = df_final.get('p75', 0)
        
        if 'p95' in base_features:
            result_df['p95'] = df_final.get('p95', 0)
        
        if 'gap_medio' in base_features:
            result_df['gap_medio'] = df_final.get('gap_medio', 0)
        
        if 'gap_max' in base_features:
            result_df['gap_max'] = df_final.get('gap_max', 0)
        
        if 'taxa_dias_ativos' in base_features:
            result_df['taxa_dias_ativos'] = np.where(
                df_final['total_dias'] > 0, 
                df_final.get('dias_ativos', 0) / df_final['total_dias'], 
                0
            )
        
        if 'taxa_semanas_ativas' in base_features:
            result_df['taxa_semanas_ativas'] = np.where(
                df_final['total_semanas'] > 0, 
                df_final.get('semanas_ativas', 0) / df_final['total_semanas'], 
                0
            )
        
        if 'cv' in base_features:
            result_df['cv'] = np.where(
                df_final.get('media', 0) > 0, 
                df_final.get('std', 0) / df_final.get('media', 1), 
                0
            )
        
        if 'cv_gaps' in base_features:
            result_df['cv_gaps'] = np.where(
                df_final.get('gap_medio', 0) > 0, 
                df_final.get('gap_std', 0) / df_final.get('gap_medio', 1), 
                0
            )
        
        if 'iqr' in base_features:
            result_df['iqr'] = df_final.get('p75', 0) - df_final.get('p25', 0)
        
        if 'score_continuidade' in base_features:
            taxa_semanas = np.where(
                df_final['total_semanas'] > 0, 
                df_final.get('semanas_ativas', 0) / df_final['total_semanas'], 
                0
            )
            cv_gaps = np.where(
                df_final.get('gap_medio', 0) > 0, 
                df_final.get('gap_std', 0) / df_final.get('gap_medio', 1), 
                0
            )
            result_df['score_continuidade'] = taxa_semanas * (1 / (1 + cv_gaps))
        
        result_df = result_df.replace([np.inf, -np.inf], 999999).fillna(0)
        
        # Adicionar prefixo 'seg_' a todas as colunas (exceto veiculo_id)
        rename_map = {col: f'{self.PREFIX}_{col}' for col in result_df.columns if col != 'veiculo_id'}
        result_df = result_df.rename(columns=rename_map)
        
        # Adicionar sufixo de métrica se necessário
        if self.add_suffix:
            rename_map = {col: f'{col}_{self.metric}' for col in result_df.columns if col != 'veiculo_id'}
            result_df = result_df.rename(columns=rename_map)
        
        return result_df[['veiculo_id'] + self.feature_names]


# ============================================================================
# WEEKDAY FEATURE EXTRACTOR
# ============================================================================

class WeekdayFeatureExtractor(BaseFeatureExtractor):
    """
    Extrator de Features por Dia da Semana
    Prefixo: 'day_' (já embutido no padrão day_1_mean, etc.)
    """
    
    PREFIX = 'day'
    NORM_RULES = {
        'mean': 'by_upper', 'std': 'by_upper', 'median': 'by_upper',
        'p25': 'by_upper', 'p75': 'by_upper', 'iqr': 'by_upper',
        'prob_active': 'no_norm',
        'cv': 'log1p',
        'max': 'by_upper', 'min': 'by_upper',
    }

    def __init__(
        self, 
        metric: Literal['km', 'h'], 
        features: Optional[List[str]] = None,
        add_suffix: bool = True
    ):
        """
        Parameters
        ----------
        metric : Literal['km', 'h']
            Métrica a ser analisada
        features : Optional[List[str]]
            Features canônicas base: ['mean', 'std', 'p25', ...]
            Cada uma gera 7 features (uma por dia)
        add_suffix : bool
            Se True, adiciona sufixo da métrica (padrão: True)
        """
        self.metric = metric
        self.add_suffix = add_suffix
        target_col = f'{metric}_dia_clean'
        
        self._base_features = [
            'mean', 'std', 'median', 'max', 'min', 
            'p25', 'p75', 'iqr', 'prob_active', 'cv'
        ]
        
        if features is None:
            self._selected_base = self._base_features.copy()
        else:
            invalid = [f for f in features if f not in self._base_features]
            if invalid:
                raise ValueError(f"Features canônicas inválidas: {invalid}")
            self._selected_base = features
        
        # Expandir (prefixo 'day_' já está no padrão)
        self._expanded_features = []
        for day in range(1, 8):
            for feat in self._selected_base:
                self._expanded_features.append(f'{self.PREFIX}_{day}_{feat}')
        
        if self.add_suffix:
            self._final_features = [f'{feat}_{self.metric}' for feat in self._expanded_features]
        else:
            self._final_features = self._expanded_features.copy()
        
        all_features = []
        for day in range(1, 8):
            for feat in self._base_features:
                all_features.append(f'{self.PREFIX}_{day}_{feat}')
        
        name = f'weekday_features_{metric}'
        super().__init__(name, all_features, required_targets=[target_col])
        
        self._feature_names = self._final_features
    
    @property
    def selected_features(self) -> List[str]:
        """Features canônicas selecionadas pelo usuário"""
        return self._selected_base.copy()
    
    @property
    def feature_names(self) -> List[str]:
        """Nomes finais das features (expandidas + sufixo)"""
        return self._final_features.copy()
    
    @property
    def feature_map(self) -> Dict[str, List[str]]:
        """Mapeamento 1:N (cada feature base gera 7)"""
        result = {}
        for feat_base in self._selected_base:
            expanded_list = []
            for day in range(1, 8):
                expanded = f'day_{day}_{feat_base}'
                if self.add_suffix:
                    expanded = f'{expanded}_{self.metric}'
                expanded_list.append(expanded)
            result[feat_base] = expanded_list
        return result

    def _needs_weekday(self) -> bool:
        return True
    
    def _get_required_base_features(self) -> set:
        base_features = set(self._selected_base)
        if 'iqr' in base_features:
            base_features.add('p25')
            base_features.add('p75')
        if 'cv' in base_features:
            base_features.add('mean')
            base_features.add('std')
        return base_features
    
    def _extract_features(self, df_pl: pl.DataFrame) -> pd.DataFrame:
        """Extrai features por dia da semana"""
        target = f'{self.metric}_dia_clean'
        
        df_pl = df_pl.filter(pl.col('dentro'))
        
        required_base = self._get_required_base_features()
        
        agg_exprs = []
        
        for day in range(1, 8):
            day_filter = pl.col('weekday') == day
            
            for feat in required_base:
                if feat == 'mean':
                    agg_exprs.append(
                        pl.col(target).filter(day_filter).mean().alias(f'{self.PREFIX}_{day}_mean')
                    )
                elif feat == 'std':
                    agg_exprs.append(
                        pl.col(target).filter(day_filter).std().alias(f'{self.PREFIX}_{day}_std')
                    )
                elif feat == 'median':
                    agg_exprs.append(
                        pl.col(target).filter(day_filter).median().alias(f'{self.PREFIX}_{day}_median')
                    )
                elif feat == 'max':
                    agg_exprs.append(
                        pl.col(target).filter(day_filter).max().alias(f'{self.PREFIX}_{day}_max')
                    )
                elif feat == 'min':
                    agg_exprs.append(
                        pl.col(target).filter(day_filter).min().alias(f'{self.PREFIX}_{day}_min')
                    )
                elif feat == 'p25':
                    agg_exprs.append(
                        pl.col(target).filter(day_filter).quantile(0.25).alias(f'{self.PREFIX}_{day}_p25')
                    )
                elif feat == 'p75':
                    agg_exprs.append(
                        pl.col(target).filter(day_filter).quantile(0.75).alias(f'{self.PREFIX}_{day}_p75')
                    )
                elif feat == 'prob_active':
                    agg_exprs.append(
                        (pl.col(target).filter(day_filter) > 0).mean().alias(f'{self.PREFIX}_{day}_prob_active')
                    )
        
        features = df_pl.group_by('veiculo_id').agg(agg_exprs)
        features = features.fill_null(0)
        df_final = features.to_pandas()
        
        result_df = df_final[['veiculo_id']].copy()
        
        for day in range(1, 8):
            if 'iqr' in self._selected_base:
                p25_col = f'{self.PREFIX}_{day}_p25'
                p75_col = f'{self.PREFIX}_{day}_p75'
                if p25_col in df_final.columns and p75_col in df_final.columns:
                    result_df[f'{self.PREFIX}_{day}_iqr'] = df_final[p75_col] - df_final[p25_col]
            
            if 'cv' in self._selected_base:
                mean_col = f'{self.PREFIX}_{day}_mean'
                std_col = f'{self.PREFIX}_{day}_std'
                if mean_col in df_final.columns and std_col in df_final.columns:
                    result_df[f'{self.PREFIX}_{day}_cv'] = np.where(
                        df_final[mean_col] > 0,
                        df_final[std_col] / df_final[mean_col],
                        0
                    )
            
            for feat in self._selected_base:
                if feat not in ['iqr', 'cv']:
                    col_name = f'{self.PREFIX}_{day}_{feat}'
                    if col_name in df_final.columns:
                        result_df[col_name] = df_final[col_name]
        
        result_df = result_df.replace([np.inf, -np.inf], 0).fillna(0)
        
        if self.add_suffix:
            rename_map = {
                col: f'{col}_{self.metric}' 
                for col in result_df.columns 
                if col != 'veiculo_id'
            }
            result_df = result_df.rename(columns=rename_map)
        
        return result_df[['veiculo_id'] + self.feature_names]


# ============================================================================
# MONTH PHASE FEATURE EXTRACTOR
# ============================================================================

class MonthPhaseFeatureExtractor(BaseFeatureExtractor):
    """
    Extrator de features por FASE DO MÊS CALENDÁRIO REAL
    Prefixo: 'phase_'
    """
    
    PREFIX = 'phase'
    NORM_RULES = {
        'mean': 'by_upper', 'p75': 'by_upper', 'iqr': 'by_upper',
        'prob_active': 'no_norm',
        'cv': 'log1p',
    }
    
    def __init__(
        self,
        metric: Literal['km', 'h'],
        features: Optional[List[str]] = None,
        add_suffix: bool = True
    ):
        """
        Parameters
        ----------
        metric : Literal['km', 'h']
            Métrica a ser analisada
        features : Optional[List[str]]
            Features canônicas: ['prob_active', 'mean', 'p75', 'iqr', 'cv']
            Cada uma gera 3 features (uma por fase: inicio, meio, fim)
        add_suffix : bool
            Se True, adiciona sufixo da métrica (padrão: True)
        """
        self.metric = metric
        self.add_suffix = add_suffix
        target_col = f'{metric}_dia_clean'
        
        self._base_features = ['prob_active', 'mean', 'p75', 'iqr', 'cv']
        self._phases = ['inicio', 'meio', 'fim']
        
        if features is None:
            self._selected_base = self._base_features.copy()
        else:
            invalid = [f for f in features if f not in self._base_features]
            if invalid:
                raise ValueError(f"Features canônicas inválidas: {invalid}")
            self._selected_base = features
        
        self._expanded_features = []
        for feat in self._selected_base:
            for phase in self._phases:
                self._expanded_features.append(f'{self.PREFIX}_{feat}_{phase}')
        
        if self.add_suffix:
            self._final_features = [f'{feat}_{self.metric}' for feat in self._expanded_features]
        else:
            self._final_features = self._expanded_features.copy()
        
        all_features = []
        for feat in self._base_features:
            for phase in self._phases:
                all_features.append(f'{self.PREFIX}_{feat}_{phase}')
        
        name = f'month_phase_features_{metric}'
        super().__init__(name, all_features, required_targets=[target_col])
        
        self._feature_names = self._final_features
    
    @property
    def selected_features(self) -> List[str]:
        """Features canônicas selecionadas pelo usuário"""
        return self._selected_base.copy()
    
    @property
    def feature_names(self) -> List[str]:
        """Nomes finais das features (expandidas + prefixo + sufixo)"""
        return self._final_features.copy()
    
    @property
    def feature_map(self) -> Dict[str, List[str]]:
        """Mapeamento 1:N (cada feature base gera 3)"""
        result = {}
        for feat_base in self._selected_base:
            expanded_list = []
            for phase in self._phases:
                expanded = f'{self.PREFIX}_{feat_base}_{phase}'
                if self.add_suffix:
                    expanded = f'{expanded}_{self.metric}'
                expanded_list.append(expanded)
            result[feat_base] = expanded_list
        return result

    def _get_required_base_features(self) -> set:
        base_features = set(self._selected_base)
        if 'iqr' in base_features:
            base_features.add('p25')
            base_features.add('p75')
        if 'cv' in base_features:
            base_features.add('mean')
            base_features.add('std')
        return base_features

    def _extract_features(self, df_pl: pl.DataFrame) -> pd.DataFrame:
        """Extrai features por fase do mês"""
        target = f'{self.metric}_dia_clean'
        
        df_pl = df_pl.filter(pl.col('dentro'))
        
        df_pl = df_pl.with_columns(
            pl.when(pl.col('data').dt.day() <= 10)
            .then(pl.lit('inicio'))
            .when(pl.col('data').dt.day() <= 20)
            .then(pl.lit('meio'))
            .otherwise(pl.lit('fim'))
            .alias('fase')
        )
        
        required_base = self._get_required_base_features()
        
        agg_exprs = []
        
        if 'prob_active' in required_base:
            agg_exprs.append((pl.col(target) > 0).mean().alias('prob_active'))
        if 'mean' in required_base:
            agg_exprs.append(pl.col(target).mean().alias('mean'))
        if 'std' in required_base:
            agg_exprs.append(pl.col(target).std().alias('std'))
        if 'p25' in required_base:
            agg_exprs.append(pl.col(target).quantile(0.25).alias('p25'))
        if 'p75' in required_base:
            agg_exprs.append(pl.col(target).quantile(0.75).alias('p75'))
        
        agg = (
            df_pl.group_by(['veiculo_id', 'fase'])
            .agg(agg_exprs)
        ).to_pandas()
        
        all_vehicles = agg['veiculo_id'].unique()
        result = pd.DataFrame({'veiculo_id': all_vehicles})
        
        for feat_expanded in self._expanded_features:
            result[feat_expanded] = 0.0
        
        for phase in self._phases:
            sub = agg[agg['fase'] == phase]
            
            if len(sub) == 0:
                continue
            
            sub = sub.set_index('veiculo_id')
            
            for feat in self._selected_base:
                if feat not in ['iqr', 'cv']:
                    col_name = f'{self.PREFIX}_{feat}_{phase}'
                    if col_name in self._expanded_features and feat in sub.columns:
                        values = sub[feat].reindex(result['veiculo_id']).fillna(0).values
                        result[col_name] = values
            
            if 'iqr' in self._selected_base:
                col_name = f'{self.PREFIX}_iqr_{phase}'
                if col_name in self._expanded_features:
                    if 'p25' in sub.columns and 'p75' in sub.columns:
                        iqr_values = (sub['p75'] - sub['p25']).reindex(result['veiculo_id']).fillna(0).values
                        result[col_name] = iqr_values
            
            if 'cv' in self._selected_base:
                col_name = f'{self.PREFIX}_cv_{phase}'
                if col_name in self._expanded_features:
                    if 'mean' in sub.columns and 'std' in sub.columns:
                        mean_values = sub['mean'].reindex(result['veiculo_id']).fillna(0).values
                        std_values = sub['std'].reindex(result['veiculo_id']).fillna(0).values
                        result[col_name] = np.where(mean_values > 0, std_values / mean_values, 0)
        
        result = result.replace([np.inf, -np.inf], 0).fillna(0)
        
        if self.add_suffix:
            rename_map = {
                col: f'{col}_{self.metric}'
                for col in result.columns
                if col != 'veiculo_id'
            }
            result = result.rename(columns=rename_map)
        
        return result[['veiculo_id'] + self.feature_names]


# ============================================================================
# MONTHLY FEATURE EXTRACTOR
# ============================================================================

class MonthlyFeatureExtractor(BaseFeatureExtractor):
    """
    Extrator de features mensais por CICLO OPERACIONAL (28 dias)
    Prefixo: 'cycle_'
    """
    
    PREFIX = 'cycle'
    NORM_RULES = {
        'mean': 'by_upper', 'prob_active': 'no_norm',
        'ratio_fim_inicio': 'ratio', 'assimetria': 'clamp',
    }
    
    def __init__(
        self,
        metric: Literal['km', 'h'],
        features: Optional[List[str]] = None,
        add_suffix: bool = True
    ):
        """
        Parameters
        ----------
        metric : Literal['km', 'h']
            Métrica a ser analisada
        features : Optional[List[str]]
            Features canônicas:
            - 'prob_active': gera 3 features por fase
            - 'mean': gera 3 features por fase
            - 'ratio_fim_inicio': feature única
            - 'assimetria': feature única
        add_suffix : bool
            Se True, adiciona sufixo da métrica (padrão: True)
        """
        self.metric = metric
        self.add_suffix = add_suffix
        target_col = f'{metric}_dia_clean'
        
        self._base_features = ['prob_active', 'mean', 'ratio_fim_inicio', 'assimetria']
        self._phases = ['inicio', 'meio', 'fim']
        self._phase_features = ['prob_active', 'mean']
        self._single_features = ['ratio_fim_inicio', 'assimetria']
        
        if features is None:
            self._selected_base = self._base_features.copy()
        else:
            invalid = [f for f in features if f not in self._base_features]
            if invalid:
                raise ValueError(f"Features canônicas inválidas: {invalid}")
            self._selected_base = features
        
        self._expanded_features = []
        for feat in self._selected_base:
            if feat in self._phase_features:
                for phase in self._phases:
                    self._expanded_features.append(f'{self.PREFIX}_{feat}_{phase}')
            else:
                self._expanded_features.append(f'{self.PREFIX}_{feat}')
        
        if self.add_suffix:
            self._final_features = [f'{feat}_{self.metric}' for feat in self._expanded_features]
        else:
            self._final_features = self._expanded_features.copy()
        
        all_features = []
        for feat in self._base_features:
            if feat in self._phase_features:
                for phase in self._phases:
                    all_features.append(f'{self.PREFIX}_{feat}_{phase}')
            else:
                all_features.append(f'{self.PREFIX}_{feat}')
        
        name = f'monthly_cycle_features_{metric}'
        super().__init__(name, all_features, required_targets=[target_col])
        
        self._feature_names = self._final_features
    
    @property
    def selected_features(self) -> List[str]:
        """Features canônicas selecionadas pelo usuário"""
        return self._selected_base.copy()
    
    @property
    def feature_names(self) -> List[str]:
        """Nomes finais das features (expandidas + prefixo + sufixo)"""
        return self._final_features.copy()
    
    @property
    def feature_map(self) -> Dict[str, Union[str, List[str]]]:
        """Mapeamento misto: algumas 1:N, outras 1:1"""
        result = {}
        for feat_base in self._selected_base:
            if feat_base in self._phase_features:
                expanded_list = []
                for phase in self._phases:
                    expanded = f'{self.PREFIX}_{feat_base}_{phase}'
                    if self.add_suffix:
                        expanded = f'{expanded}_{self.metric}'
                    expanded_list.append(expanded)
                result[feat_base] = expanded_list
            else:
                expanded = f'{self.PREFIX}_{feat_base}'
                if self.add_suffix:
                    expanded = f'{expanded}_{self.metric}'
                result[feat_base] = expanded
        return result

    def _extract_features(self, df_pl: pl.DataFrame) -> pd.DataFrame:
        """Extrai features mensais"""
        target = f'{self.metric}_dia_clean'
        
        df_pl = df_pl.filter(pl.col('dentro'))
        
        df_pl = df_pl.with_columns(
            pl.int_range(1, pl.len() + 1).over('veiculo_id').alias('dia_idx')
        )
        
        df_pl = df_pl.with_columns([
            ((pl.col('dia_idx') - 1) // 28).alias('ciclo_id'),
            ((pl.col('dia_idx') - 1) % 28 + 1).alias('pos_ciclo')
        ])
        
        df_pl = df_pl.with_columns(
            pl.when(pl.col('pos_ciclo') <= 9)
            .then(pl.lit('inicio'))
            .when(pl.col('pos_ciclo') <= 18)
            .then(pl.lit('meio'))
            .otherwise(pl.lit('fim'))
            .alias('fase')
        )
        
        agg_exprs = []
        
        if 'prob_active' in self._selected_base:
            agg_exprs.append((pl.col(target) > 0).mean().alias('prob_active'))
        
        if any(f in self._selected_base for f in ['mean', 'ratio_fim_inicio', 'assimetria']):
            agg_exprs.append(pl.col(target).mean().alias('mean_val'))
        
        agg = (
            df_pl.group_by(['veiculo_id', 'fase'])
            .agg(agg_exprs)
        ).to_pandas()
        
        result = pd.DataFrame({'veiculo_id': agg['veiculo_id'].unique()})
        
        if 'prob_active' in agg.columns:
            prob = agg.pivot(index='veiculo_id', columns='fase', values='prob_active')
        else:
            prob = pd.DataFrame(index=result['veiculo_id'])
        
        if 'mean_val' in agg.columns:
            mean = agg.pivot(index='veiculo_id', columns='fase', values='mean_val')
        else:
            mean = pd.DataFrame(index=result['veiculo_id'])
        
        for phase in self._phases:
            if phase not in prob.columns:
                prob[phase] = 0.0
            if phase not in mean.columns:
                mean[phase] = 0.0
        
        prob = prob.fillna(0.0)
        mean = mean.fillna(0.0)
        
        for phase in self._phases:
            if 'prob_active' in self._selected_base:
                col_name = f'{self.PREFIX}_prob_active_{phase}'
                if col_name in self._expanded_features:
                    result[col_name] = prob[phase].reindex(result['veiculo_id']).fillna(0).values
            
            if 'mean' in self._selected_base:
                col_name = f'{self.PREFIX}_mean_{phase}'
                if col_name in self._expanded_features:
                    result[col_name] = mean[phase].reindex(result['veiculo_id']).fillna(0).values
        
        if 'ratio_fim_inicio' in self._selected_base:
            mean_inicio = mean['inicio'].reindex(result['veiculo_id']).fillna(0).values
            mean_fim = mean['fim'].reindex(result['veiculo_id']).fillna(0).values
            result[f'{self.PREFIX}_ratio_fim_inicio'] = np.where(mean_inicio > 0, mean_fim / mean_inicio, 0)
        
        if 'assimetria' in self._selected_base:
            mean_inicio = mean['inicio'].reindex(result['veiculo_id']).fillna(0).values
            mean_meio = mean['meio'].reindex(result['veiculo_id']).fillna(0).values
            mean_fim = mean['fim'].reindex(result['veiculo_id']).fillna(0).values
            total = mean_inicio + mean_meio + mean_fim + 1e-6
            result[f'{self.PREFIX}_assimetria'] = (mean_fim - mean_inicio) / total
        
        result = result.replace([np.inf, -np.inf], 0).fillna(0)
        
        if self.add_suffix:
            rename_map = {
                col: f'{col}_{self.metric}'
                for col in result.columns
                if col != 'veiculo_id'
            }
            result = result.rename(columns=rename_map)
        
        return result[['veiculo_id'] + self.feature_names]


# ============================================================================
# FACTORY
# ============================================================================

_EXTRACTOR_REGISTRY = {
    'type': TypeFeatureExtractor,
    'seg': SegmentationFeatureExtractor,
    'day': WeekdayFeatureExtractor,
    'phase': MonthPhaseFeatureExtractor,
    'cycle': MonthlyFeatureExtractor,
}


def get_extractor_by_prefix(
    prefix: str,
    features: Optional[List[str]] = None,
    metric: Optional[str] = None,
    add_suffix: Optional[bool] = None,
) -> BaseFeatureExtractor:
    """
    Cria um extractor pelo prefixo.

    Parameters
    ----------
    prefix : str
        Prefixo do extractor: 'type', 'seg', 'day', 'phase', 'cycle'.
    features : Optional[List[str]]
        Features canônicas a extrair (None = todas).
    metric : Optional[str]
        Métrica ('km' ou 'h'). Obrigatória para todos exceto 'type'.
    add_suffix : Optional[bool]
        Se True, adiciona sufixo da métrica. Não se aplica a 'type'.
        None = True (padrão).
    """
    cls = _EXTRACTOR_REGISTRY.get(prefix)
    if cls is None:
        raise ValueError(
            f"Prefixo desconhecido: '{prefix}'. "
            f"Válidos: {list(_EXTRACTOR_REGISTRY.keys())}"
        )

    if prefix == 'type':
        return cls(features=features)

    if metric is None:
        raise ValueError(f"metric é obrigatório para o extractor '{prefix}'")

    suffix = add_suffix if add_suffix is not None else True
    return cls(metric=metric, features=features, add_suffix=suffix)


def get_norm_rules(prefix: str) -> Dict[str, str]:
    """Retorna as NORM_RULES do extrator com o prefixo correspondente."""
    cls = _EXTRACTOR_REGISTRY.get(prefix)
    if cls is None:
        raise ValueError(
            f"Prefixo desconhecido: '{prefix}'. "
            f"Válidos: {list(_EXTRACTOR_REGISTRY.keys())}"
        )
    return cls.NORM_RULES
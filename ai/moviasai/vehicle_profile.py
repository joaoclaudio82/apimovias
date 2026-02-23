import polars as pl
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Union, Optional, Dict, List, Tuple
from datetime import datetime, timedelta
import warnings
from joblib import Parallel, delayed
from tqdm import tqdm


class VehicleProfile:
    """
    Perfil completo de veículos com versionamento semanal
    
    Mantém dois DataFrames:
    1. df_versions: versões semanais de todas as features
    2. df_effective_period: período efetivo versionado
    
    E um dicionário:
    3. target_samples: amostras por veículo para update incremental
    """
    
    def __init__(
        self,
        target: str,
        sample_size: int = 364,
        p_upper: int = 99,
        n_jobs: int = -1
    ):
        """
        Inicializa o perfil de veículos
        
        Parameters:
        -----------
        target : str
            Nome da coluna alvo
        sample_size : int, optional
            Número máximo de amostras (padrão: 364)
        p_upper : int, optional
            Percentil superior para clipping (padrão: 99)
        n_jobs : int, optional
            Número de processos paralelos. -1 usa todos os cores (padrão: -1)
        """
        self.target = target
        self.metric = target.split('_')[0]
        self.sample_size = sample_size
        self.p_upper = p_upper
        self.n_jobs = n_jobs
        
        self.df_versions: Optional[pd.DataFrame] = None
        self.df_effective_period: Optional[pd.DataFrame] = None
        self.target_samples: Dict[int, List[float]] = {}
        
        self._setup_feature_names()
    
    def _setup_feature_names(self):
        """Define nomes das features"""
        m = self.metric
        
        self.segmentation_features = [
            f'{m}_por_dia', f'media_{m}', f'max_{m}', f'mediana_{m}', f'std_{m}',
            f'score_continuidade_{m}', f'taxa_semanas_ativas_{m}', f'taxa_dias_ativos_{m}',
            f'cv_gaps_{m}', f'gap_medio_{m}', f'gap_max_{m}', f'cv_{m}',
            f'p25_{m}', f'p75_{m}', f'iqr_{m}'
        ]
        
        self.weekday_base_features = [
            'mean', 'std', 'median', 'max', 'min', 'p25', 'p75', 'iqr', 'prob_active', 'cv'
        ]
        
        self.weekday_columns = []
        for day in range(1, 8):
            for feat in self.weekday_base_features:
                self.weekday_columns.append(f'day_{day}_{feat}')
        
        self.scaler_columns = ['upper']
    
    def _get_week_end_date(self, date: Union[str, datetime]) -> datetime:
        """Retorna domingo"""
        if isinstance(date, str):
            date = pd.to_datetime(date, format='%d/%m/%Y')
        days_to_sunday = 6 - date.weekday()
        return date + timedelta(days=days_to_sunday)
    
    def _get_effective_period(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calcula período efetivo"""
        return (
            df.filter(pl.col(self.target) > 0)
            .with_columns([
                pl.col('data').dt.year().alias('ano'),
                pl.col('data').dt.week().alias('semana')
            ])
            .group_by(['veiculo_id', 'ano', 'semana'])
            .agg(pl.col('data').min().alias('primeira_data_semana'))
            .group_by('veiculo_id')
            .agg([
                pl.col('primeira_data_semana').min().alias('primeira'),
                pl.col('primeira_data_semana').max().alias('ultima')
            ])
            .with_columns([
                (pl.col('primeira') - pl.duration(days=pl.col('primeira').dt.weekday() - 1)).alias('dt_inicio'),
                (pl.col('ultima') + pl.duration(days=7 - pl.col('ultima').dt.weekday())).alias('dt_fim')
            ])
            .select(['veiculo_id', 'dt_inicio', 'dt_fim'])
        )
    
    def _extract_segmentation_features(
        self,
        df: pl.DataFrame,
        up_to_date: datetime
    ) -> pd.DataFrame:
        """Extrai features de segmentação"""
        target = self.target
        df = df.filter(pl.col('data') <= up_to_date)
        
        df_pl = df.with_columns([
            pl.col('data').dt.year().alias('ano'),
            pl.col('data').dt.week().alias('semana')
        ])
        
        periodo = self._get_effective_period(df_pl)
        df_pl = df_pl.join(periodo, on='veiculo_id', how='left')
        
        df_pl = df_pl.with_columns([
            ((pl.col('data') >= pl.col('dt_inicio')) & 
             (pl.col('data') <= pl.col('dt_fim'))).alias('dentro')
        ])
        
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
        
        features = agg.join(gaps, on='veiculo_id', how='left').fill_null(0)
        df_final = features.to_pandas()
        
        m = self.metric
        df_final[f'{m}_por_dia'] = np.where(df_final['total_dias'] > 0, df_final['total'] / df_final['total_dias'], 0)
        df_final[f'media_{m}'] = df_final['media']
        df_final[f'max_{m}'] = df_final['max']
        df_final[f'mediana_{m}'] = df_final['mediana']
        df_final[f'std_{m}'] = df_final['std']
        df_final[f'p25_{m}'] = df_final['p25']
        df_final[f'p75_{m}'] = df_final['p75']
        df_final[f'gap_medio_{m}'] = df_final['gap_medio']
        df_final[f'gap_max_{m}'] = df_final['gap_max']
        df_final[f'taxa_dias_ativos_{m}'] = np.where(df_final['total_dias'] > 0, df_final['dias_ativos'] / df_final['total_dias'], 0)
        df_final[f'taxa_semanas_ativas_{m}'] = np.where(df_final['total_semanas'] > 0, df_final['semanas_ativas'] / df_final['total_semanas'], 0)
        df_final[f'cv_{m}'] = np.where(df_final['media'] > 0, df_final['std'] / df_final['media'], 0)
        df_final[f'cv_gaps_{m}'] = np.where(df_final['gap_medio'] > 0, df_final['gap_std'] / df_final['gap_medio'], 0)
        df_final[f'iqr_{m}'] = df_final[f'p75_{m}'] - df_final[f'p25_{m}']
        df_final[f'score_continuidade_{m}'] = df_final[f'taxa_semanas_ativas_{m}'] * (1 / (1 + df_final[f'cv_gaps_{m}']))
        
        df_final = df_final.replace([np.inf, -np.inf], 999999).fillna(0)
        
        return df_final[['veiculo_id'] + self.segmentation_features]
    
    def _extract_weekday_features(
        self,
        df: pl.DataFrame,
        up_to_date: datetime
    ) -> pd.DataFrame:
        """Extrai features por dia da semana"""
        target = self.target
        df = df.filter(pl.col('data') <= up_to_date)
        
        df_wd = df.with_columns([
            pl.col('data').dt.weekday().alias('weekday')
        ])
        
        agg = (
            df_wd
            .group_by(['veiculo_id', 'weekday'])
            .agg([
                pl.col(target).mean().alias('mean'),
                pl.col(target).std().alias('std'),
                pl.col(target).median().alias('median'),
                pl.col(target).max().alias('max'),
                pl.col(target).min().alias('min'),
                pl.col(target).quantile(0.25).alias('p25'),
                pl.col(target).quantile(0.75).alias('p75'),
                (pl.col(target) > 0).mean().alias('prob_active')
            ])
        )
        
        df_agg = agg.to_pandas()
        df_agg['iqr'] = df_agg['p75'] - df_agg['p25']
        df_agg['cv'] = np.where(df_agg['mean'] > 0, df_agg['std'] / df_agg['mean'], 0)
        df_agg = df_agg.fillna(0)
        
        df_wide = df_agg.pivot(
            index='veiculo_id',
            columns='weekday',
            values=self.weekday_base_features
        )
        
        df_wide.columns = [f'day_{int(day)}_{feat}' for feat, day in df_wide.columns]
        df_wide = df_wide.reset_index()
        
        for day in range(1, 8):
            for feat in self.weekday_base_features:
                col = f'day_{day}_{feat}'
                if col not in df_wide.columns:
                    df_wide[col] = 0.0
        
        return df_wide
    
    @staticmethod
    def _process_week_version(
        year: int,
        week: int,
        df_pl: pl.DataFrame,
        target: str,
        p_upper: int,
        sample_size: int,
        segmentation_features: List[str],
        weekday_columns: List[str],
        extract_seg_fn,
        extract_wd_fn,
        get_period_fn
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, Tuple[float, List[float]]]]:
        """Processa uma semana em paralelo"""
        df_week = df_pl.with_columns([
            pl.col('data').dt.year().alias('y'),
            pl.col('data').dt.week().alias('w')
        ]).filter((pl.col('y') == year) & (pl.col('w') == week))
        
        if len(df_week) == 0:
            return None, None, {}
        
        week_end = df_week['data'].max()
        df_cumulative = df_pl.filter(pl.col('data') <= week_end)
        
        df_seg = extract_seg_fn(df_cumulative, week_end)
        df_wd = extract_wd_fn(df_cumulative, week_end)
        df_period = get_period_fn(df_cumulative).to_pandas()
        
        df_combined = df_seg.merge(df_wd, on='veiculo_id', how='inner')
        df_combined['year'] = year
        df_combined['week'] = week
        
        df_period['year'] = year
        df_period['week'] = week
        
        scaler_data = {}
        for vehicle_id in df_combined['veiculo_id']:
            df_v = df_cumulative.filter(pl.col('veiculo_id') == vehicle_id).sort('data')
            values = df_v[target].to_numpy()[-sample_size:]
            
            upper = float(np.percentile(values, p_upper)) if len(values) > 0 else 1.0
            scaler_data[vehicle_id] = (upper, values.tolist())
        
        df_combined['upper'] = df_combined['veiculo_id'].map(lambda v: scaler_data[v][0])
        
        return df_combined, df_period, scaler_data
    
    def fit(self, df: Union[pl.DataFrame, pd.DataFrame], verbose: bool = True):
        """Ajusta perfis com paralelização"""
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
            print(f"AJUSTANDO PERFIS - MODO PARALELO ({self.metric.upper()})")
            print(f"{'='*80}")
            print(f"Target: {self.target}")
            print(f"Veículos: {df_pl['veiculo_id'].n_unique()}")
            print(f"Período: {df_pl['data'].min()} a {df_pl['data'].max()}")
            print(f"Processos: {self.n_jobs if self.n_jobs > 0 else 'todos'}")
            print()
        
        df_with_week = df_pl.with_columns([
            pl.col('data').dt.year().alias('year'),
            pl.col('data').dt.week().alias('week')
        ])
        
        all_weeks = (
            df_with_week
            .select(['year', 'week'])
            .unique()
            .sort(['year', 'week'])
        ).to_pandas()
        
        if verbose:
            print(f"📅 Total de semanas: {len(all_weeks)}")
            print("🔄 Processando em paralelo...")
        
        import time
        start_time = time.time()
        
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._process_week_version)(
                int(row['year']),
                int(row['week']),
                df_pl,
                self.target,
                self.p_upper,
                self.sample_size,
                self.segmentation_features,
                self.weekday_columns,
                self._extract_segmentation_features,
                self._extract_weekday_features,
                self._get_effective_period
            )
            for _, row in tqdm(all_weeks.iterrows(), total=len(all_weeks), desc="Semanas", disable=not verbose)
        )
        
        valid_results = [(v, p, s) for v, p, s in results if v is not None]
        
        all_versions = [r[0] for r in valid_results]
        all_periods = [r[1] for r in valid_results]
        
        for _, _, scaler_data in valid_results:
            for vehicle_id, (upper, samples) in scaler_data.items():
                self.target_samples[vehicle_id] = samples
        
        self.df_versions = pd.concat(all_versions, ignore_index=True)
        self.df_effective_period = pd.concat(all_periods, ignore_index=True)
        
        base_cols = ['year', 'week', 'veiculo_id']
        feature_cols = self.segmentation_features + self.weekday_columns + self.scaler_columns
        self.df_versions = self.df_versions[base_cols + feature_cols]
        
        period_cols = ['year', 'week', 'veiculo_id', 'dt_inicio', 'dt_fim']
        self.df_effective_period = self.df_effective_period[period_cols]
        
        elapsed = time.time() - start_time
        
        if verbose:
            print(f"✓ Concluído em {elapsed:.2f}s")
            print(f"  - Versões: {len(self.df_versions)}")
            print(f"  - Veículos: {self.df_versions['veiculo_id'].nunique()}")
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
                for feat in self.segmentation_features:
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
        """
        Busca perfis para múltiplas combinações (veículo, data) em BATCH
        
        Parameters:
        -----------
        vehicle_ids : List[int]
            Lista de IDs de veículos
        dates : List[datetime]
            Lista de datas
        normalized : bool, optional
            Se True, normaliza features
        
        Returns:
        --------
        dict
            {(vehicle_id, date): profile_row}
        """
        # Converter datas
        dates_dt = []
        for date in dates:
            if isinstance(date, str):
                dates_dt.append(pd.to_datetime(date, format='%d/%m/%Y'))
            else:
                dates_dt.append(date)
        
        # Mapear (date -> year, week)
        date_to_week = {}
        for date in dates_dt:
            week_end = self._get_week_end_date(date)
            year = week_end.year
            week = week_end.isocalendar()[1]
            date_to_week[date] = (year, week)
        
        # Obter semanas únicas
        unique_weeks = list(set(date_to_week.values()))
        
        # Filtrar df_versions para semanas e veículos necessários
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
        
        # Criar cache
        profiles_cache = {}
        
        for date in dates_dt:
            target_year, target_week = date_to_week[date]
            
            for vid in vehicle_ids:
                # Buscar versão exata
                exact = df_filtered[
                    (df_filtered['veiculo_id'] == vid) &
                    (df_filtered['year'] == target_year) &
                    (df_filtered['week'] == target_week)
                ]
                
                if len(exact) > 0:
                    row = exact.iloc[0].copy()
                else:
                    # Buscar última anterior
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
                
                # Normalizar se necessário
                if normalized:
                    upper = row['upper']
                    for feat in self.segmentation_features:
                        if feat in row:
                            row[feat] = np.clip(row[feat], 0, upper) / upper if upper > 0 else 0
                
                profiles_cache[(vid, date)] = row
        
        return profiles_cache
    
    def get_vehicle_static_covariates(
        self,
        vehicle_id: int,
        date: Union[str, datetime],
        normalized: bool = False
    ) -> pd.DataFrame:
        """Retorna features de segmentação"""
        profile_df = self.get_profile(vehicle_id, date, normalized=normalized)
        
        if len(profile_df) == 0:
            raise ValueError(f"Perfil não encontrado")
        
        return profile_df[self.segmentation_features].iloc[0:1]
    
    def get_daily_features(
        self,
        vehicle_id: int,
        input_dates: pd.DatetimeIndex,
        reference_date: Union[str, datetime]
    ) -> np.ndarray:
        """Retorna features diárias para um vetor de datas"""
        profile_df = self.get_profile(vehicle_id, reference_date)
        
        if len(profile_df) == 0:
            return np.zeros((len(input_dates), len(self.weekday_base_features)))
        
        profile_row = profile_df.iloc[0]
        
        n_dates = len(input_dates)
        n_features = len(self.weekday_base_features)
        features_matrix = np.zeros((n_dates, n_features))
        
        for i, date in enumerate(input_dates):
            weekday = date.weekday() + 1
            
            for j, feat in enumerate(self.weekday_base_features):
                col_name = f'day_{weekday}_{feat}'
                features_matrix[i, j] = profile_row[col_name]
        
        return features_matrix
    
    def transform(
        self,
        vehicle_id: Union[int, List[int]],
        values: Union[np.ndarray, List[np.ndarray]],
        date: Union[str, datetime]
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Normaliza com clip [0, upper]"""
        if isinstance(vehicle_id, int):
            profile = self.get_profile(vehicle_id, date)
            upper = profile['upper'].values[0]
            clipped = np.clip(values, 0, upper)
            return clipped / upper if upper > 0 else np.zeros_like(clipped)
        else:
            profiles = self.get_profile(vehicle_id, date)
            scaled_list = []
            for i, vid in enumerate(vehicle_id):
                upper = profiles[profiles['veiculo_id'] == vid]['upper'].values[0]
                clipped = np.clip(values[i], 0, upper)
                scaled_list.append(clipped / upper if upper > 0 else np.zeros_like(clipped))
            return scaled_list
    
    def inverse_transform(
        self,
        vehicle_id: Union[int, List[int]],
        scaled_values: Union[np.ndarray, List[np.ndarray]],
        date: Union[str, datetime]
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Desnormaliza"""
        if isinstance(vehicle_id, int):
            profile = self.get_profile(vehicle_id, date)
            upper = profile['upper'].values[0]
            return scaled_values * upper
        else:
            profiles = self.get_profile(vehicle_id, date)
            denorm_list = []
            for i, vid in enumerate(vehicle_id):
                upper = profiles[profiles['veiculo_id'] == vid]['upper'].values[0]
                denorm_list.append(scaled_values[i] * upper)
            return denorm_list
    
    def save(self, output_dir: str):
        """Salva perfis"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.df_versions.to_parquet(output_path / 'vehicle_profiles.parquet', index=False)
        self.df_effective_period.to_parquet(output_path / 'effective_periods.parquet', index=False)
        
        with open(output_path / 'target_samples.json', 'w') as f:
            json.dump(self.target_samples, f)
        
        metadata = {
            'target': self.target,
            'metric': self.metric,
            'sample_size': self.sample_size,
            'p_upper': self.p_upper,
            'n_vehicles': int(self.df_versions['veiculo_id'].nunique()),
            'n_versions': len(self.df_versions)
        }
        
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Salvos em: {output_path}")
    
    @classmethod
    def load(cls, profile_dir: str) -> 'VehicleProfile':
        """Carrega perfis"""
        output_path = Path(profile_dir)
        
        with open(output_path / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        instance = cls(
            target=metadata['target'],
            sample_size=metadata['sample_size'],
            p_upper=metadata['p_upper']
        )
        
        instance.df_versions = pd.read_parquet(output_path / 'vehicle_profiles.parquet')
        instance.df_effective_period = pd.read_parquet(output_path / 'effective_periods.parquet')
        
        with open(output_path / 'target_samples.json', 'r') as f:
            instance.target_samples = {int(k): v for k, v in json.load(f).items()}
        
        print(f"✓ Carregado: {metadata['n_vehicles']} veículos, {metadata['n_versions']} versões")
        
        return instance
    
    def __repr__(self) -> str:
        n_versions = len(self.df_versions) if self.df_versions is not None else 0
        n_vehicles = self.df_versions['veiculo_id'].nunique() if self.df_versions is not None else 0
        
        return (
            f"VehicleProfile(\n"
            f"  target='{self.target}',\n"
            f"  n_vehicles={n_vehicles},\n"
            f"  n_versions={n_versions},\n"
            f"  n_jobs={self.n_jobs}\n"
            f")"
        )
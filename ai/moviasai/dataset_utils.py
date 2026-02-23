import polars as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import json

from datetime import timedelta, datetime
from pathlib import Path

from typing import Optional, Tuple, Dict, Literal, Union
from scipy.stats import gaussian_kde
from scipy.signal import argrelextrema


import time

import warnings
from collections import Counter


def next_day(dt: datetime, weekday=0) -> datetime:
    """
    Retorna a próxima ocorrência de um dia da semana
    
    Parameters:
    -----------
    dt : datetime
        Data de referência
    weekday : int, optional
        Dia da semana (0=Segunda, 6=Domingo). Padrão: 0 (Segunda)
        
    Returns:
    --------
    datetime
        Próxima ocorrência do dia da semana (ou a própria data se já for o dia)
    """
    days_until_weekday = (weekday - dt.weekday() + 7) % 7
    if days_until_weekday == 0:
        return dt
    else:
        return dt + timedelta(days=days_until_weekday)


def previous_day(dt: datetime, weekday=6) -> datetime:
    """
    Retorna a ocorrência anterior de um dia da semana
    
    Parameters:
    -----------
    dt : datetime
        Data de referência
    weekday : int, optional
        Dia da semana (0=Segunda, 6=Domingo). Padrão: 6 (Domingo)
        
    Returns:
    --------
    datetime
        Ocorrência anterior do dia da semana (ou a própria data se já for o dia)
    """
    days_since_weekday = (dt.weekday() - weekday + 7) % 7
    if days_since_weekday == 0:
        return dt
    else:
        return dt - timedelta(days=days_since_weekday)


class DatasetSlicer:
    """
    Construtor de janelas temporais com critério de semanas ativas (OTIMIZADO)
    
    Cria janelas temporais onde TODAS as semanas devem ter atividade.
    Uma semana é considerada ativa se tem pelo menos um dia com target > 0.
    
    Usa processamento vetorizado para máxima performance.
    
    Attributes:
    -----------
    target : str
        Nome da coluna alvo
    window_size : int
        Tamanho da janela em dias (deve ser múltiplo de 7)
    stride : int
        Passo entre janelas consecutivas (em dias)
    max_gap : int or None
        Máximo de dias de gap permitido entre registros
    n_weeks : int
        Número de semanas na janela (calculado automaticamente)
    """
    
    def __init__(
        self,
        target: str,
        window_size: int,
        stride: int,
        max_gap: Optional[int] = 7
    ):
        """
        Inicializa o construtor de janelas
        
        Parameters:
        -----------
        target : str
            Nome da coluna alvo
        window_size : int
            Tamanho da janela em dias (deve ser múltiplo de 7)
        stride : int
            Passo entre janelas consecutivas (em dias)
        max_gap : int or None, optional
            Máximo de dias de gap permitido (padrão: 7)
            Se None, usa períodos efetivos
        
        Raises:
        -------
        ValueError
            Se window_size não for múltiplo de 7
        
        Examples:
        ---------
        >>> slicer = DatasetSlicer(
        ...     target='km_dia_clean',
        ...     window_size=35,  # 5 semanas
        ...     stride=7
        ... )
        """
        self.target = target
        self.window_size = window_size
        self.stride = stride
        self.max_gap = max_gap
        
        # Validar que window_size é múltiplo de 7
        if window_size % 7 != 0:
            raise ValueError(
                f"window_size deve ser múltiplo de 7 (semanas completas). "
                f"Recebido: {window_size}"
            )
        
        # Calcular número de semanas na janela
        self.n_weeks = window_size // 7
    
    def _get_effective_period(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calcula período efetivo baseado em semanas com atividade
        
        Returns:
        --------
        pl.DataFrame
            Colunas: veiculo_id, inicio_periodo, fim_periodo
        """
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
                (pl.col('primeira') - pl.duration(days=pl.col('primeira').dt.weekday() - 1)).alias('inicio_periodo'),
                (pl.col('ultima') + pl.duration(days=7 - pl.col('ultima').dt.weekday())).alias('fim_periodo')
            ])
            .select(['veiculo_id', 'inicio_periodo', 'fim_periodo'])
        )
    
    def _identify_effective_periods(self, df_valid: pl.DataFrame) -> pl.DataFrame:
        """Identifica períodos efetivos (quando max_gap=None)"""
        periodos = self._get_effective_period(df_valid)
        
        return (
            periodos
            .with_columns([
                pl.lit(0).alias('id_periodo')
            ])
            .rename({
                'inicio_periodo': 'data_inicio',
                'fim_periodo': 'data_fim'
            })
        )
    
    def _identify_gap_limited_periods(self, df_valid: pl.DataFrame) -> pl.DataFrame:
        """Identifica períodos com gaps limitados"""
        return (
            df_valid
            .with_columns([
                pl.col('data').diff().dt.total_days().fill_null(1).alias('day_diff')
            ])
            .with_columns([
                (pl.col('day_diff') > self.max_gap).cast(pl.Int32).alias('new_period')
            ])
            .with_columns([
                pl.col('new_period').cum_sum().over('veiculo_id').alias('id_periodo')
            ])
            .group_by(['veiculo_id', 'id_periodo'])
            .agg([
                pl.col('data').min().alias('data_inicio_seg'),
                pl.col('data').max().alias('data_fim_seg')
            ])
        )
    
    def _align_periods_to_weeks(self, periods: pl.DataFrame) -> pl.DataFrame:
        """Alinha períodos para semanas completas"""
        return periods.with_columns([
            # Alinhar início para segunda-feira
            (
                pl.col("data_inicio_seg") +
                pl.when(pl.col("data_inicio_seg").dt.weekday() != 1)
                .then(
                    ((8 - pl.col("data_inicio_seg").dt.weekday()) % 7).cast(pl.Int32) * pl.duration(days=1)
                )
                .otherwise(pl.duration(days=0))
            ).alias("data_inicio"),
            # Alinhar fim para domingo
            (
                pl.col("data_fim_seg") -
                pl.when(pl.col("data_fim_seg").dt.weekday() != 7)
                .then((pl.col("data_fim_seg").dt.weekday() % 7).cast(pl.Int32) * pl.duration(days=1))
                .otherwise(pl.duration(days=0))
            ).alias("data_fim"),
        ])
    
    def _calculate_period_days(self, periods: pl.DataFrame) -> pl.DataFrame:
        """Calcula número de dias de cada período"""
        return (
            periods
            .with_columns([
                (pl.col('data_fim') - pl.col('data_inicio')).dt.total_days().alias('num_dias_diff')
            ])
            .with_columns([
                (pl.col('num_dias_diff') + 1).alias('num_dias')
            ])
        )
    
    def _count_records_in_periods(
        self,
        periods: pl.DataFrame,
        df: pl.DataFrame
    ) -> pl.DataFrame:
        """Conta registros reais dentro de cada período"""
        aligned = periods.join(
            df.filter(pl.col(self.target) > 0).select(['veiculo_id', 'data']),
            on='veiculo_id',
            how='left'
        ).filter(
            (pl.col('data') >= pl.col('data_inicio')) & (pl.col('data') <= pl.col('data_fim'))
        )
        
        return (
            aligned
            .group_by(['veiculo_id', 'id_periodo', 'data_inicio', 'data_fim', 'num_dias'])
            .agg([
                pl.col('data').n_unique().alias('qtd_registros')
            ])
            .select([
                'veiculo_id', 'id_periodo', 'data_inicio', 'data_fim', 'num_dias', 'qtd_registros'
            ])
        )
    
    def _identify_continuous_periods(self, df: pl.DataFrame) -> pl.DataFrame:
        """Identifica períodos contínuos"""
        df_valid = df.filter(pl.col(self.target) > 0).sort(['veiculo_id', 'data'])
        
        if self.max_gap is None:
            periods = self._identify_effective_periods(df_valid)
        else:
            periods = self._identify_gap_limited_periods(df_valid)
            periods = self._align_periods_to_weeks(periods)
        
        periods = self._calculate_period_days(periods)
        periods = periods.filter(pl.col('num_dias') >= self.window_size)
        result = self._count_records_in_periods(periods, df)
        
        return result
    
    def _precompute_active_weeks(
        self,
        df: pl.DataFrame,
        verbose: bool = False
    ) -> pl.DataFrame:
        """
        Pré-computa todas as semanas ativas para todos os veículos
        
        Parameters:
        -----------
        df : pl.DataFrame
            DataFrame com dados
        verbose : bool, optional
            Se True, imprime logs
        
        Returns:
        --------
        pl.DataFrame
            DataFrame com semanas ativas:
            - veiculo_id
            - ano
            - semana
            - week_start (segunda-feira)
            - week_end (domingo)
        """
        if verbose:
            print("   🔄 Pré-computando semanas ativas...")
        
        start = time.time()
        
        # Filtrar apenas registros com atividade
        df_active = df.filter(pl.col(self.target) > 0)
        
        # Adicionar informações de semana
        df_weeks = (
            df_active
            .with_columns([
                pl.col('data').dt.year().alias('ano'),
                pl.col('data').dt.week().alias('semana')
            ])
            .group_by(['veiculo_id', 'ano', 'semana'])
            .agg([
                pl.col('data').min().alias('first_date')
            ])
            .with_columns([
                # Calcular segunda-feira da semana
                (pl.col('first_date') - pl.duration(days=pl.col('first_date').dt.weekday() - 1)).alias('week_start'),
            ])
            .with_columns([
                # Calcular domingo da semana
                (pl.col('week_start') + pl.duration(days=6)).alias('week_end')
            ])
            .select(['veiculo_id', 'ano', 'semana', 'week_start', 'week_end'])
        )
        
        if verbose:
            elapsed = time.time() - start
            n_weeks = len(df_weeks)
            n_vehicles = df_weeks['veiculo_id'].n_unique()
            print(f"      ✓ {n_weeks:,} semanas ativas para {n_vehicles:,} veículos ({elapsed:.2f}s)")
        
        return df_weeks
    
    def _extract_windows_ts(
        self,
        df: pl.DataFrame,
        df_windows: pl.DataFrame
    ) -> pl.DataFrame:
        """Extrai dados das janelas"""
        df_join = df.join(
            df_windows,
            on='veiculo_id',
            how='inner'
        )
        
        df_filtered = df_join.filter(
            (pl.col('data') >= pl.col('data_inicio_window')) &
            (pl.col('data') <= pl.col('data_fim_window'))
        )
        
        return df_filtered.select([
            'veiculo_id', 'window', 'data', self.target
        ]).sort(['veiculo_id', 'window', 'data'])
    
    def _get_segments(
        self,
        df: pl.DataFrame,
        df_periods: pl.DataFrame,
        verbose: bool = False
    ) -> pl.DataFrame:
        """
        Gera segmentos de janelas com processamento VETORIZADO
        
        Critério: TODAS as semanas da janela devem ter atividade
        
        Parameters:
        -----------
        df : pl.DataFrame
            DataFrame original
        df_periods : pl.DataFrame
            Períodos contínuos
        verbose : bool, optional
            Se True, imprime logs
        
        Returns:
        --------
        pl.DataFrame
            Segmentos de janelas válidas
        """
        if verbose:
            print("\n" + "="*70)
            print("🔍 GERAÇÃO DE JANELAS (VETORIZADA)")
            print("="*70)
        
        start_time = time.time()
        
        # 1. PRÉ-COMPUTAR SEMANAS ATIVAS
        df_active_weeks = self._precompute_active_weeks(df, verbose)
        
        # 2. Calcular limites das janelas
        min_date = df_periods['data_inicio'].min()
        max_date = df_periods['data_fim'].max()
        total_days = (max_date - min_date).days + 1
        
        if verbose:
            print(f"\n📅 Período disponível:")
            print(f"   min_date: {min_date}")
            print(f"   max_date: {max_date}")
            print(f"   total_days: {total_days}")
            print(f"\n📏 Parâmetros:")
            print(f"   window_size: {self.window_size} dias ({self.n_weeks} semanas)")
            print(f"   stride: {self.stride} dias")
            print(f"   Critério: TODAS as {self.n_weeks} semanas devem ter atividade")
        
        if total_days < self.window_size:
            if verbose:
                print(f"\n⚠️  Período muito curto ({total_days} < {self.window_size})")
            return pl.DataFrame(schema={
                'veiculo_id': pl.Int64,
                'window': pl.Int64,
                'data_inicio_window': pl.Date,
                'data_fim_window': pl.Date
            })
        
        # 3. Gerar todas as janelas possíveis
        n_windows = ((total_days - self.window_size) // self.stride) + 1
        
        if verbose:
            print(f"\n📆 Gerando {n_windows} janelas...")
        
        windows_list = []
        for i in range(n_windows):
            window_end = max_date - timedelta(days=self.stride * i)
            window_start = window_end - timedelta(days=self.window_size - 1)
            
            if window_start < min_date:
                continue
            
            windows_list.append({
                'window': n_windows - 1 - i,  # Reverter ordem para cronológica
                'data_inicio_window': window_start,
                'data_fim_window': window_end
            })
        
        df_windows = pl.DataFrame(windows_list)
        
        if verbose:
            print(f"   ✓ {len(df_windows)} janelas válidas (dentro do período)")
        
        # 4. CROSS JOIN: janelas × veículos
        if verbose:
            print("\n   🔄 Criando combinações janela × veículo...")
        
        df_candidates = df_windows.join(
            df_periods.select(['veiculo_id', 'data_inicio', 'data_fim']),
            how='cross'
        )
        
        if verbose:
            print(f"      ✓ {len(df_candidates):,} combinações criadas")
        
        # 5. FILTRAR: janela deve estar completamente dentro do período do veículo
        if verbose:
            print("   🔄 Filtrando por período do veículo...")
        
        df_candidates = df_candidates.filter(
            (pl.col('data_inicio_window') >= pl.col('data_inicio')) &
            (pl.col('data_fim_window') <= pl.col('data_fim'))
        )
        
        if verbose:
            print(f"      ✓ {len(df_candidates):,} candidatos após filtro de período")
        
        # 6. JOIN com semanas ativas
        if verbose:
            print("   🔄 Verificando semanas ativas...")
        
        df_with_weeks = df_candidates.join(
            df_active_weeks,
            on='veiculo_id',
            how='left'
        )
        
        # 7. FILTRAR: semanas que estão dentro da janela
        df_with_weeks = df_with_weeks.filter(
            (pl.col('week_start') >= pl.col('data_inicio_window')) &
            (pl.col('week_end') <= pl.col('data_fim_window'))
        )
        
        # 8. CONTAR semanas ativas por (veículo, janela)
        df_week_counts = (
            df_with_weeks
            .group_by(['veiculo_id', 'window', 'data_inicio_window', 'data_fim_window'])
            .agg([
                pl.col('semana').n_unique().alias('n_active_weeks')
            ])
        )
        
        if verbose:
            print(f"      ✓ Semanas contadas para {len(df_week_counts):,} combinações")
        
        # 9. FILTRAR: apenas onde TODAS as semanas estão ativas
        if verbose:
            print(f"   🔄 Filtrando por critério de {self.n_weeks} semanas ativas...")
        
        df_valid = df_week_counts.filter(
            pl.col('n_active_weeks') == self.n_weeks
        ).select(['veiculo_id', 'window', 'data_inicio_window', 'data_fim_window'])
        
        elapsed = time.time() - start_time
        
        if verbose:
            print(f"      ✓ {len(df_valid):,} segmentos válidos")
            print(f"\n   ⏱️  Tempo total de processamento: {elapsed:.2f}s")
        
        # 10. Estatísticas
        if verbose and len(df_valid) > 0:
            print("\n" + "="*70)
            print("📊 RESUMO")
            print("="*70)
            print(f"   Tempo de processamento: {elapsed:.2f}s")
            print(f"   Janelas processadas: {n_windows}")
            print(f"   Segmentos válidos: {len(df_valid):,}")
            
            vehicles_per_window = (
                df_valid
                .group_by('window')
                .agg(pl.col('veiculo_id').n_unique().alias('n_veiculos'))
                .sort('window')
            )
            
            avg_vehicles = vehicles_per_window['n_veiculos'].mean()
            min_vehicles = vehicles_per_window['n_veiculos'].min()
            max_vehicles = vehicles_per_window['n_veiculos'].max()
            
            print(f"\n   Distribuição de veículos por janela:")
            print(f"      Média: {avg_vehicles:.0f} veículos")
            print(f"      Min: {min_vehicles} veículos")
            print(f"      Max: {max_vehicles} veículos")
            
            # Mostrar detalhes se poucas janelas
            if len(vehicles_per_window) <= 20:
                print(f"\n   Detalhamento:")
                for row in vehicles_per_window.iter_rows(named=True):
                    print(f"      Window {row['window']}: {row['n_veiculos']} veículos")
            
            print("="*70 + "\n")
        
        return df_valid
    
    def slice(
        self,
        df: pl.DataFrame,
        verbose: bool = False
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Constrói janelas de dados
        
        Pipeline:
        1. Identifica períodos contínuos
        2. Pré-computa semanas ativas (vetorizado)
        3. Gera segmentos validando todas as semanas (vetorizado)
        4. Extrai dados das janelas
        
        Parameters:
        -----------
        df : pl.DataFrame
            DataFrame com colunas: veiculo_id, data, {target}
        verbose : bool, optional
            Se True, imprime logs detalhados (padrão: False)
        
        Returns:
        --------
        tuple
            (df_windows, df_periods)
            - df_windows: Dados organizados em janelas
            - df_periods: Metadados dos períodos contínuos
        
        Examples:
        ---------
        >>> slicer = DatasetSlicer(
        ...     target='km_dia_clean',
        ...     window_size=35,
        ...     stride=7
        ... )
        >>> df_windows, df_periods = slicer.slice(df, verbose=True)
        """
        if verbose:
            print("\n🚀 Iniciando construção de janelas...")
            print(f"   Target: {self.target}")
            print(f"   Window size: {self.window_size} dias ({self.n_weeks} semanas)")
            print(f"   Stride: {self.stride} dias")
            print(f"   Max gap: {self.max_gap if self.max_gap is not None else 'None (período efetivo)'}")
        
        total_start = time.time()
        
        # 1. Identificar períodos contínuos
        if verbose:
            print("\n📍 Etapa 1: Identificando períodos contínuos...")
        
        df_periods = self._identify_continuous_periods(df)
        
        if verbose:
            print(f"   ✓ {len(df_periods):,} períodos identificados")
            print(f"   ✓ {df_periods['veiculo_id'].n_unique():,} veículos únicos")
        
        # 2. Gerar segmentos de janelas (VETORIZADO)
        if verbose:
            print("\n📍 Etapa 2: Gerando segmentos de janelas (vetorizado)...")
        
        df_dataset_segments = self._get_segments(
            df=df,
            df_periods=df_periods,
            verbose=verbose
        )
        
        # 3. Extrair dados das janelas
        if verbose:
            print("\n📍 Etapa 3: Extraindo dados das janelas...")
        
        extract_start = time.time()
        df_windows = self._extract_windows_ts(df, df_dataset_segments)
        extract_elapsed = time.time() - extract_start
        
        if verbose:
            print(f"   ✓ {len(df_windows):,} registros extraídos ({extract_elapsed:.2f}s)")
        
        total_elapsed = time.time() - total_start
        
        if verbose:
            print(f"\n✅ Construção de janelas concluída em {total_elapsed:.2f}s!\n")
        
        return df_windows, df_periods
    
    def __repr__(self) -> str:
        max_gap_str = f"{self.max_gap}" if self.max_gap is not None else "None (período efetivo)"
        
        return (
            f"DatasetSlicer(\n"
            f"  target='{self.target}',\n"
            f"  window_size={self.window_size} ({self.n_weeks} semanas),\n"
            f"  stride={self.stride},\n"
            f"  max_gap={max_gap_str}\n"
            f")"
        )


class DatasetFormatter:
    """
    Formatador de datasets de séries temporais
    
    Esta classe processa dados brutos de veículos, identifica períodos
    contínuos, aplica filtros de qualidade e preenche datas ausentes.
    
    Attributes:
    -----------
    df : polars.DataFrame
        DataFrame com dados brutos
    target : str
        Nome da coluna alvo
    min_days : int
        Número mínimo de dias para considerar um período válido
    min_weeks : int
        Número mínimo de semanas para veículos válidos
    max_gap : int
        Máximo de dias de gap permitido entre registros
    """
    
    def __init__(
        self, 
        dataset_path: str, 
        target: str, 
        min_days: int = 7, 
        min_weeks: int = 5, 
        max_gap: int = 3
    ):
        """
        Inicializa o formatador de dataset
        
        Parameters:
        -----------
        dataset_path : str or Path
            Caminho do arquivo CSV com dados brutos
        target : str
            Nome da coluna alvo (ex: 'km_dia_clean')
        min_days : int, optional
            Número mínimo de dias para períodos válidos (padrão: 7)
        min_weeks : int, optional
            Número mínimo de semanas para veículos válidos (padrão: 5)
        max_gap : int, optional
            Máximo de dias de gap permitido (padrão: 3)
        """
        self.df = pl.read_csv(
            dataset_path,
            columns=['veiculo_id', 'data', target],
            try_parse_dates=True
        )
        
        self.target = target
        self.min_days = min_days
        self.min_weeks = min_weeks
        self.max_gap = max_gap
        
        print(f"Dataset carregado: {len(self.df):,} registros")
        print(f"Veiculos: {self.df['veiculo_id'].n_unique()}")
        print(f"Target: {target}")
        print()
    
    def _get_df_week(self) -> pl.DataFrame:
        """
        Calcula médias diárias por semana 
        
        Este método processa os dados brutos e calcula a média diária
        por semana para cada veículo.
        
        Returns:
        --------
        polars.DataFrame
            DataFrame com colunas:
            - veiculo_id: ID do veículo
            - ano_semana: Ano e semana (formato: YYYY-Wnn)
            - media_dia: Média diária da semana
        """
        # Processar dados
        df_proc = self.df.select(['veiculo_id', 'data', self.target]).unique()
        df_proc = df_proc.sort(['veiculo_id', 'data'])
        
        # Criar coluna ano_semana
        df_proc = df_proc.with_columns([
            (pl.col('data').dt.year().cast(pl.Utf8) + 
             '-W' + 
             pl.col('data').dt.week().cast(pl.Utf8).str.zfill(2)
            ).alias('ano_semana')
        ])
        
        # Filtrar apenas valores válidos
        df_proc = df_proc.filter(pl.col(self.target) > 0)
        
        # Calcular média diária por semana
        df_week = (
            df_proc.group_by(['veiculo_id', 'ano_semana'])
            .agg([
                (pl.col(self.target).sum() / pl.col('data').count()).alias('media_dia')
            ])
        )
        
        # Filtrar veículos com pelo menos min_weeks semanas
        veiculos_validos = (
            df_week.group_by('veiculo_id')
            .agg(pl.col('ano_semana').count().alias('n_semanas'))
            .filter(pl.col('n_semanas') >= self.min_weeks)
            .select('veiculo_id')
        )
        
        df_week = df_week.join(veiculos_validos, on='veiculo_id', how='inner')
        
        return df_week
    
    def _get_df_stats(
        self, 
        stat_metric: Literal['max', 'mean', 'median', 'min', 'std'] = 'max'
    ) -> pl.DataFrame:
        """
        Calcula estatísticas por veículo (REUTILIZA _get_df_week)
        
        Parameters:
        -----------
        stat_metric : str, optional
            Métrica estatística a calcular:
            - 'max': Máximo (padrão)
            - 'mean': Média
            - 'median': Mediana
            - 'min': Mínimo
            - 'std': Desvio padrão
            
        Returns:
        --------
        polars.DataFrame
            DataFrame com colunas:
            - veiculo_id: ID do veículo
            - stat_value: Valor da estatística escolhida
        """
        df_week = self._get_df_week()
        
        # Mapear métrica para função Polars
        metric_map = {
            'max': pl.col('media_dia').max(),
            'mean': pl.col('media_dia').mean(),
            'median': pl.col('media_dia').median(),
            'min': pl.col('media_dia').min(),
            'std': pl.col('media_dia').std()
        }
        
        if stat_metric not in metric_map:
            raise ValueError(
                f"stat_metric invalido: {stat_metric}. "
                f"Opcoes: {list(metric_map.keys())}"
            )
        
        # Calcular estatística
        df_stats = (
            df_week.group_by('veiculo_id')
            .agg([
                metric_map[stat_metric].alias('stat_value')
            ])
        )
        
        return df_stats
    
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
            pl.col(self.target).fill_null(0)
        ])
        
        # 7. Ordenar
        df_filled = df_filled.sort(['veiculo_id', 'data'])
        
        # Estatísticas
        n_zeros = (df_filled[self.target] == 0).sum()
        pct_zeros = n_zeros / len(df_filled) * 100
        
        print(f"  • Registros com zero: {n_zeros:,} ({pct_zeros:.1f}%)")
        print()
        
        return df_filled
    
    def visualize_distribution(
        self,
        stat_metric: Literal['max', 'mean', 'median', 'min', 'std'] = 'max',
        save_plots: bool = True
    ) -> Dict:
        """
        Visualiza a distribuição de uma métrica estatística
        
        Cria visualizações da distribuição usando histograma, KDE,
        e identifica picos e vales automaticamente.
        
        Parameters:
        -----------
        stat_metric : str, optional
            Métrica estatística a visualizar: 'max', 'mean', 'median', 'min', 'std'
            (padrão: 'max')
        save_plots : bool, optional
            Se True, salva visualizações (padrão: True)
            
        Returns:
        --------
        dict
            Dicionário com informações da análise:
            - stat_metric: Métrica usada
            - n_vehicles: Número de veículos
            - min_value: Valor mínimo
            - max_value: Valor máximo
            - mean_value: Valor médio
            - median_value: Valor mediano
            - peaks: Lista com valores dos picos
            - valleys: Lista com valores dos vales
            - subdistributions: Lista com info de cada subdistribuição
            - vehicle_stats: Series com valores por veículo
        """
        print("=" * 80)
        print(f"VISUALIZACAO: DISTRIBUICAO DE {stat_metric.upper()}")
        print("=" * 80)
        print()
        
        df_week = self._get_df_week()
        df_week_pandas = df_week.to_pandas()
        
        # Calcular estatística escolhida
        stat_map = {
            'max': lambda x: x.max(),
            'mean': lambda x: x.mean(),
            'median': lambda x: x.median(),
            'min': lambda x: x.min(),
            'std': lambda x: x.std()
        }
        
        vehicle_stats = df_week_pandas.groupby('veiculo_id')['media_dia'].agg(stat_map[stat_metric])
        vehicle_stats = vehicle_stats.dropna()
        
        # Estatísticas básicas
        print(f"Estatisticas de {stat_metric.upper()}:")
        print(f"  • N veiculos: {len(vehicle_stats)}")
        print(f"  • Min: {vehicle_stats.min():.2f}")
        print(f"  • Max: {vehicle_stats.max():.2f}")
        print(f"  • Media: {vehicle_stats.mean():.2f}")
        print(f"  • Mediana: {vehicle_stats.median():.2f}")
        print(f"  • Desvio: {vehicle_stats.std():.2f}")
        print()
        
        # Percentis com quantidade de veículos
        print("Percentis (valor e quantidade acumulada de veiculos):")
        print(f"{'Percentil':>10s} | {'Valor':>10s} | {'N Veiculos':>12s} | {'% Acumulado':>14s}")
        print("-" * 52)
        
        for p in range(1, 100): #[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 25, 50, 75, 90, 95, 99]:
            value = vehicle_stats.quantile(p/100)
            n_vehicles = (vehicle_stats <= value).sum()
            pct_acum = n_vehicles / len(vehicle_stats) * 100
            print(f"P{p:2d}        | {value:10.2f} | {n_vehicles:12d} | {pct_acum:13.1f}%")
        print()
        
        # Calcular KDE
        kde = gaussian_kde(vehicle_stats)
        x_range = np.linspace(vehicle_stats.min(), vehicle_stats.max(), 1000)
        density = kde(x_range)
        
        # Identificar picos e vales
        local_maxima_idx = argrelextrema(density, np.greater, order=50)[0]
        local_minima_idx = argrelextrema(density, np.less, order=50)[0]
        
        peaks = [x_range[idx] for idx in local_maxima_idx]
        valleys = [x_range[idx] for idx in local_minima_idx]
        
        # Análise de picos (subdistribuições)
        print(f"Picos identificados: {len(peaks)}")
        print(f"{'Pico':>6s} | {'Valor':>10s} | {'N Proximos':>12s} | {'Media':>10s} | {'Desvio':>10s} | {'Range':>20s}")
        print("-" * 78)
        
        subdistributions = []
        for i, peak in enumerate(peaks[:5], 1):
            # Definir janela ao redor do pico (±2 desvios ou até próximo vale)
            if i == 1:
                # Primeiro pico: do mínimo até o primeiro vale (ou pico+margem)
                lower_bound = vehicle_stats.min()
                if len(valleys) > 0 and valleys[0] > peak:
                    upper_bound = valleys[0]
                else:
                    upper_bound = peak * 1.5
            elif i <= len(valleys):
                # Picos intermediários: entre vales
                lower_bound = valleys[i-2] if i > 1 else vehicle_stats.min()
                upper_bound = valleys[i-1]
            else:
                # Último pico: do último vale até o máximo
                lower_bound = valleys[-1]
                upper_bound = vehicle_stats.max()
            
            # Dados da subdistribuição
            subdist_data = vehicle_stats[(vehicle_stats >= lower_bound) & (vehicle_stats <= upper_bound)]
            
            if len(subdist_data) > 0:
                n_subdist = len(subdist_data)
                mean_subdist = subdist_data.mean()
                std_subdist = subdist_data.std()
                min_subdist = subdist_data.min()
                max_subdist = subdist_data.max()
                
                subdistributions.append({
                    'peak': peak,
                    'n_vehicles': n_subdist,
                    'mean': mean_subdist,
                    'std': std_subdist,
                    'min': min_subdist,
                    'max': max_subdist
                })
                
                print(f"P{i:2d}    | {peak:10.2f} | {n_subdist:12d} | {mean_subdist:10.2f} | {std_subdist:10.2f} | [{min_subdist:.1f}, {max_subdist:.1f}]")
        print()
        
        print(f"Vales identificados: {len(valleys)}")
        print(f"{'Vale':>6s} | {'Valor':>10s} | {'N Abaixo':>10s} | {'% Abaixo':>10s} | {'N Acima':>10s} | {'% Acima':>10s}")
        print("-" * 64)
        
        for i, valley in enumerate(valleys[:5], 1):
            n_below = (vehicle_stats < valley).sum()
            pct_below = n_below / len(vehicle_stats) * 100
            n_above = (vehicle_stats >= valley).sum()
            pct_above = n_above / len(vehicle_stats) * 100
            print(f"V{i:2d}    | {valley:10.2f} | {n_below:10d} | {pct_below:9.1f}% | {n_above:10d} | {pct_above:9.1f}%")
        print()
        
        # Visualizações
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        preffix = self.target.split('_')[0]
        unit_label = f'({preffix}/dia)'
        
        # Gráfico 1: Histograma
        ax1 = axes[0, 0]
        ax1.hist(vehicle_stats, bins=100, color='#2E86AB', alpha=0.7, edgecolor='black')
        ax1.set_xlabel(f'{stat_metric.upper()} {unit_label}', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Numero de Veiculos', fontsize=12, fontweight='bold')
        ax1.set_title('Histograma', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Gráfico 2: KDE com picos e vales
        ax2 = axes[0, 1]
        ax2.plot(x_range, density, linewidth=3, color='#2E86AB', label='KDE')
        ax2.fill_between(x_range, density, alpha=0.3, color='#2E86AB')
        
        # Marcar picos
        for i, idx in enumerate(local_maxima_idx[:5], 1):
            ax2.scatter(x_range[idx], density[idx], color='red', s=200, 
                       marker='^', zorder=5, edgecolor='black', linewidth=2)
            ax2.text(x_range[idx], density[idx] + 0.001, f'P{i}\n{x_range[idx]:.1f}', 
                    ha='center', fontsize=9, fontweight='bold')
        
        # Marcar vales
        for i, idx in enumerate(local_minima_idx[:5], 1):
            ax2.scatter(x_range[idx], density[idx], color='green', s=200, 
                       marker='v', zorder=5, edgecolor='black', linewidth=2)
            ax2.text(x_range[idx], density[idx] - 0.0008, f'V{i}\n{x_range[idx]:.1f}', 
                    ha='center', fontsize=9, fontweight='bold', va='top')
        
        # Histograma ao fundo
        ax2_twin = ax2.twinx()
        ax2_twin.hist(vehicle_stats, bins=100, density=True, 
                     color='gray', alpha=0.15, edgecolor='none')
        ax2_twin.set_ylabel('Histograma (normalizado)', fontsize=10, color='gray')
        ax2_twin.tick_params(axis='y', labelcolor='gray')
        
        ax2.set_xlabel(f'{stat_metric.upper()} {unit_label}', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Densidade', fontsize=12, fontweight='bold')
        ax2.set_title('Densidade (KDE) - Picos e Vales', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Gráfico 3: Box plot
        ax3 = axes[1, 0]
        bp = ax3.boxplot([vehicle_stats], vert=False, patch_artist=True,
                         widths=0.5, showmeans=True)
        bp['boxes'][0].set_facecolor('#2E86AB')
        bp['boxes'][0].set_alpha(0.7)
        ax3.set_xlabel(f'{stat_metric.upper()} {unit_label}', fontsize=12, fontweight='bold')
        ax3.set_title('Box Plot', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Gráfico 4: Distribuição acumulada
        ax4 = axes[1, 1]
        sorted_stats = np.sort(vehicle_stats)
        cumulative = np.arange(1, len(sorted_stats) + 1) / len(sorted_stats) * 100
        ax4.plot(sorted_stats, cumulative, linewidth=2, color='#2E86AB')
        
        # Marcar vales na distribuição acumulada
        for i, valley in enumerate(valleys[:5], 1):
            pct = (vehicle_stats < valley).sum() / len(vehicle_stats) * 100
            ax4.scatter(valley, pct, color='green', s=150, zorder=5,
                       edgecolor='black', linewidth=2)
            ax4.text(valley, pct, f' V{i}', fontsize=9, fontweight='bold', va='center')
        
        ax4.set_xlabel(f'{stat_metric.upper()} {unit_label}', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Percentual Acumulado (%)', fontsize=12, fontweight='bold')
        ax4.set_title('Distribuicao Acumulada', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'DISTRIBUICAO: {stat_metric.upper()}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'resultados_segmentacao/distribuicao_{stat_metric}.png', 
                       dpi=300, bbox_inches='tight')
            print(f"Visualizacao salva: distribuicao_{stat_metric}.png")
        
        plt.show()
        print()
        
        return {
            'stat_metric': stat_metric,
            'n_vehicles': len(vehicle_stats),
            'min_value': vehicle_stats.min(),
            'max_value': vehicle_stats.max(),
            'mean_value': vehicle_stats.mean(),
            'median_value': vehicle_stats.median(),
            'std_value': vehicle_stats.std(),
            'peaks': peaks,
            'valleys': valleys,
            'subdistributions': subdistributions,
            'vehicle_stats': vehicle_stats
        }
    
    def format(
        self, 
        threshold_min: Optional[float] = None,
        threshold_max: Optional[float] = None,
        stat_metric: Optional[Literal['max', 'mean', 'median', 'min', 'std']] = 'mean',
        output_dir: Optional[str] = None,
        verbose: bool = True
    ) -> pl.DataFrame:
        """
        Formata o dataset aplicando filtros e preenchendo datas
        
        Este método executa o pipeline completo:
        1. Constrói janelas de dados contínuos
        2. Aplica filtros de threshold_min e/ou threshold_max (se especificados)
        3. Preenche datas ausentes com zero
        4. Salva o dataset (se output_dir especificado)
        
        Parameters:
        -----------
        threshold_min : float, optional
            Valor mínimo. Remove veículos com stat < threshold_min.
            Se None, não aplica filtro mínimo (padrão: None)
        threshold_max : float, optional
            Valor máximo. Remove veículos com stat >= threshold_max.
            Se None, não aplica filtro máximo (padrão: None)
        stat_metric : str, optional
            Métrica para filtro: 'max', 'mean', 'median', 'min', 'std'.
            Obrigatório se threshold_min ou threshold_max forem especificados.
            Se None e não houver thresholds, não aplica filtro (padrão: None)
        output_dir : str, optional
            Diretório para salvar o dataset. Se None, não salva (padrão: None)
        verbose : bool, optional
            Se True, imprime informações (padrão: True)
            
        Returns:
        --------
        polars.DataFrame
            DataFrame formatado com colunas:
            - veiculo_id: ID do veículo
            - data: Data do registro
            - {target}: Valor do target (0 para datas ausentes)

        Raises:
        -------
        ValueError
            Se threshold_min ou threshold_max forem especificados mas stat_metric for None
        Examples:
        ---------
        >>> formatter = DatasetFormatter('data.csv', target='km_dia_clean')
        >>> 
        >>> # Sem salvar
        >>> df = formatter.format(threshold_min=20, stat_metric='max')
        >>> 
        >>> # Salvar automaticamente
        >>> df = formatter.format(
        ...     threshold_min=20,
        ...     threshold_max=500,
        ...     stat_metric='max',
        ...     output_dir='datasets_processados'
        ... )
        """
        if verbose:
            print("=" * 80)
            print("FORMATANDO DATASET")
            print("=" * 80)
            print(f"Target: {self.target}")
            print(f"Min days: {self.min_days}")
            print(f"Min weeks: {self.min_weeks}")
            print(f"Max gap: {self.max_gap}")
            
            if threshold_min is not None or threshold_max is not None:
                if stat_metric is None:
                    raise ValueError(
                        "stat_metric é obrigatório quando threshold_min ou threshold_max são especificados"
                    )
                filter_text = f"Filtro: {stat_metric}"
                if threshold_min is not None and threshold_max is not None:
                    filter_text += f" em [{threshold_min}, {threshold_max}]"
                elif threshold_min is not None:
                    filter_text += f" >= {threshold_min}"
                elif threshold_max is not None:
                    filter_text += f" < {threshold_max}"
                print(filter_text)

            if output_dir is not None:
                print(f"Output dir: {output_dir}")
            
            print()
        
        builder = DatasetSlicer(target=self.target, stride=7, max_gap=self.max_gap, window_size=self.min_days)
        
        # 1. Construir janelas
        df_window, _ = builder.slice(
            self.df, 
            verbose=verbose
        )
        
        if verbose:
            print(f"Apos construcao de janelas:")
            print(f"  • Registros: {len(df_window):,}")
            print(f"  • Veiculos: {df_window['veiculo_id'].n_unique()}")
            print(f"  • Segmentos: {df_window['window'].n_unique()}")
            print()
        
        # 2. Aplicar filtros de threshold (se especificados)
        if threshold_min is not None or threshold_max is not None:

            df_stats = self._get_df_stats(stat_metric=stat_metric)
            
            if verbose:
                print(f"Calculando estatisticas ({stat_metric})...")
                print(f"  • Veiculos antes do filtro: {len(df_stats)}")
            
            # Aplicar filtro mínimo
            if threshold_min is not None:
                df_stats_before = df_stats
                df_stats = df_stats.filter(pl.col('stat_value') >= threshold_min)
                
                if verbose:
                    n_removed_min = len(df_stats_before) - len(df_stats)
                    pct_removed_min = n_removed_min / len(df_stats_before) * 100
                    print(f"  • Filtro minimo ({stat_metric} >= {threshold_min}):")
                    print(f"    - Veiculos removidos: {n_removed_min} ({pct_removed_min:.1f}%)")
            
            # Aplicar filtro máximo
            if threshold_max is not None:
                df_stats_before = df_stats
                df_stats = df_stats.filter(pl.col('stat_value') < threshold_max)
                
                if verbose:
                    n_removed_max = len(df_stats_before) - len(df_stats)
                    pct_removed_max = n_removed_max / len(df_stats_before) * 100
                    print(f"  • Filtro maximo ({stat_metric} <= {threshold_max}):")
                    print(f"    - Veiculos removidos: {n_removed_max} ({pct_removed_max:.1f}%)")
            
            if verbose:
                n_total_removed = df_window['veiculo_id'].n_unique() - len(df_stats)
                pct_total_removed = n_total_removed / df_window['veiculo_id'].n_unique() * 100
                print(f"  • Total apos filtros: {len(df_stats)} veiculos")
                print(f"  • Total removido: {n_total_removed} veiculos ({pct_total_removed:.1f}%)")
                print()
            
            # Filtrar df_window para veículos válidos
            veiculos_validos = df_stats['veiculo_id'].to_list()
            df_window = df_window.filter(pl.col('veiculo_id').is_in(veiculos_validos))
            
            if verbose:
                print(f"Apos filtros de threshold:")
                print(f"  • Registros: {len(df_window):,}")
                print(f"  • Veiculos: {df_window['veiculo_id'].n_unique()}")
                print()
        
        # 3. Preencher datas ausentes com zero
        if verbose:
            print("Preenchendo datas ausentes...")
        
        df_window_final = self._fill_missing_dates(df_window).select(["veiculo_id", "data", self.target])
        
        if verbose:
            n_zeros = (df_window_final[self.target] == 0).sum()
            pct_zeros = n_zeros / len(df_window_final) * 100
            
            print(f"Apos preenchimento:")
            print(f"  • Registros totais: {len(df_window_final):,}")
            print(f"  • Registros com zero: {n_zeros:,} ({pct_zeros:.1f}%)")
            print(f"  • Veiculos: {df_window_final['veiculo_id'].n_unique()}")
            print()
        
        # 4. Salvar dataset (se output_dir especificado)
        if output_dir is not None:        
            # Criar diretório se não existir
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Construir nome do arquivo
            filename_parts = [
                'dataset',
                self.target,
                f'mindays{self.min_days}',
                f'minweeks{self.min_weeks}',
                f'maxgap{self.max_gap}',
            ]

            if threshold_min is not None or threshold_max is not None:
                filename_parts.append(f'metric{stat_metric}')
            
            if threshold_min is not None:
                threshold_min_str = f"{threshold_min:.2f}".replace('.', 'p')
                filename_parts.append(f'min{threshold_min_str}')
            
            if threshold_max is not None:
                threshold_max_str = f"{threshold_max:.2f}".replace('.', 'p')
                filename_parts.append(f'max{threshold_max_str}')
            
            filename = '_'.join(filename_parts) + '.csv'
            filepath = output_path / filename
            
            # Salvar
            df_window_final.write_csv(filepath)
            
            if verbose:
                print(f"Dataset salvo:")
                print(f"  • Arquivo: {filepath}")
                print(f"  • Tamanho: {filepath.stat().st_size / (1024**2):.2f} MB")
                print()
        
        if verbose:
            print("=" * 80)
            print("DATASET FORMATADO COM SUCESSO")
            print("=" * 80)
            print()
        
        return df_window_final
    
    def get_stats(
        self, 
        stat_metric: Literal['max', 'mean', 'median', 'min', 'std'] = 'max'
    ) -> pl.DataFrame:
        """
        Retorna estatísticas dos veículos
        
        Parameters:
        -----------
        stat_metric : str, optional
            Métrica a calcular (padrão: 'max')
            
        Returns:
        --------
        polars.DataFrame
            DataFrame com estatísticas por veículo
        """
        return self._get_df_stats(stat_metric=stat_metric)
    
    def get_weekly_data(self) -> pl.DataFrame:
        """
        Retorna dados agregados por semana
        
        Returns:
        --------
        polars.DataFrame
            DataFrame com dados semanais
        """
        return self._get_df_week()
    
    def __repr__(self) -> str:
        """Representação em string do objeto"""
        return (
            f"DatasetFormatter(\n"
            f"  target='{self.target}',\n"
            f"  min_days={self.min_days},\n"
            f"  min_weeks={self.min_weeks},\n"
            f"  max_gap={self.max_gap},\n"
            f"  n_veiculos={self.df['veiculo_id'].n_unique()},\n"
            f"  n_registros={len(self.df):,}\n"
            f")"
        )
    

class DatasetGenerator:
    """
    Classe para processamento e geração de datasets de séries temporais
    
    Esta classe processa dados de veículos, identifica períodos contínuos,
    e gera janelas temporais para treino, validação e teste.
    
    Attributes:
    -----------
    df : polars.DataFrame
        DataFrame com os dados brutos
    target : str
        Nome da coluna alvo (extraído do nome do arquivo)
    history_size : int
        Tamanho da janela de histórico (em dias)
    forecast_horizon : int
        Horizonte de previsão (em dias)
    stride : int
        Passo entre janelas consecutivas (em dias)
    max_gap : int
        Máximo de dias de gap permitido (extraído do nome do arquivo)
    """
    
    def __init__(
        self, 
        dataset_path: str, 
        history_size: int, 
        forecast_horizon: int,
        stride: Optional[int] = None
    ):
        """
        Inicializa o dataset com parâmetros de janela temporal
        
        O target e max_gap são extraídos automaticamente do nome do arquivo.
        
        Parameters:
        -----------
        dataset_path : str or Path
            Caminho do arquivo CSV. O nome deve seguir o padrão:
            dataset_{target}_mindays{X}_minweeks{Y}_maxgap{Z}_...csv
        history_size : int
            Tamanho da janela de histórico (em dias)
        forecast_horizon : int
            Horizonte de previsão (em dias)
        stride : int, optional
            Passo entre janelas consecutivas (em dias).
            Se None, usa forecast_horizon (padrão: None)
        
        Examples:
        ---------
        >>> gen = DatasetGenerator(
        ...     'dataset_km_dia_clean_mindays7_minweeks5_maxgap3_metricmax_min20p50.csv',
        ...     history_size=30,
        ...     forecast_horizon=7
        ... )
        >>> # target='km_dia_clean', max_gap=3, stride=7
        
        >>> gen = DatasetGenerator(
        ...     'dataset_km_dia_clean_mindays7_minweeks5_maxgap3_metricmax_min20p50.csv',
        ...     history_size=30,
        ...     forecast_horizon=7,
        ...     stride=1
        ... )
        >>> # target='km_dia_clean', max_gap=3, stride=1
        """
        
        filepath = Path(dataset_path)
        filename = filepath.stem  # Nome sem extensão
        
        # Extrair target do nome do arquivo
        # Padrão: dataset_{target}_mindays
        match_target = re.search(r'dataset_([^_]+(?:_[^_]+)*?)_mindays', filename)
        if match_target:
            target = match_target.group(1)
            print(f"Target extraido: '{target}'")
        else:
            raise ValueError(
                f"Nao foi possivel extrair 'target' do nome do arquivo: {filename}\n"
                f"Esperado padrao: dataset_{{target}}_mindays..."
            )
        
        # Extrair max_gap do nome do arquivo
        # Padrão: ...maxgap{N}...
        match_gap = re.search(r'maxgap(\d+)', filename)
        if match_gap:
            max_gap = int(match_gap.group(1))
            print(f"Max_gap extraido: {max_gap}")
        else:
            raise ValueError(
                f"Nao foi possivel extrair 'max_gap' do nome do arquivo: {filename}\n"
                f"Esperado padrao: ...maxgap{{N}}..."
            )
        
        print()
        
        # Ler dataset
        self.df = pl.read_csv(
            dataset_path,
            columns=['veiculo_id', 'data', target],
            try_parse_dates=True
        )
        
        self.target = target
        self.history_size = history_size
        self.forecast_horizon = forecast_horizon
        self.stride = stride if stride is not None else forecast_horizon
        self.max_gap = max_gap
        
        self._dataset_name = filename
        
        print(f"Dataset carregado: {len(self.df):,} registros")
        print(f"Veiculos: {self.df['veiculo_id'].n_unique()}")
        print(f"Target: {target}")
        print(f"History size: {history_size} dias")
        print(f"Forecast horizon: {forecast_horizon} dias")
        print(f"Stride: {self.stride} dias")
        print(f"Max gap: {max_gap} dias")
        print()
    
    def _prepare_data(self, skip_n_days: int = 0) -> pl.DataFrame:
        """
        Prepara os dados removendo os últimos n dias de cada veículo
        
        Parameters:
        -----------
        skip_n_days : int, optional
            Número de dias a remover do final da série temporal de cada veículo (padrão: 0)
        
        Returns:
        --------
        polars.DataFrame
            DataFrame filtrado
        """
        df_filtered = self.df
        
        if skip_n_days > 0:
            # Calcular a data máxima para cada veículo
            max_dates = (
                self.df
                .group_by('veiculo_id')
                .agg(pl.col('data').max().alias('max_data'))
            )
            
            # Calcular a data de corte para cada veículo
            max_dates = max_dates.with_columns(
                (pl.col('max_data') - timedelta(days=skip_n_days)).alias('cutoff_date')
            )
            
            # Fazer join e filtrar
            df_filtered = (
                self.df
                .join(max_dates, on='veiculo_id', how='left')
                .filter(pl.col('data') <= pl.col('cutoff_date'))
                .select(['veiculo_id', 'data', self.target])
            )
        
        return df_filtered
    
    def _split_windows(
        self,
        df_windows: pl.DataFrame,
        n_windows_test: int,
        n_windows_val: int
    ) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Divide as janelas em treino, validação e teste
        
        A divisão é feita com base no índice da janela (window):
        - Teste: últimas n_windows_test janelas
        - Validação: n_windows_val janelas anteriores ao teste
        - Treino: todas as janelas anteriores à validação
        
        IMPORTANTE: Pode haver overlap temporal entre os splits, pois a divisão
        é feita apenas nas janelas já geradas. Isso garante consistência independente
        da configuração de n_windows_test e n_windows_val.
        
        Parameters:
        -----------
        df_windows : polars.DataFrame
            DataFrame com todas as janelas geradas
        n_windows_test : int
            Número de janelas para teste
        n_windows_val : int
            Número de janelas para validação
        
        Returns:
        --------
        tuple of polars.DataFrame
            (df_train, df_val, df_test)
        """
        # Obter o número total de janelas
        max_window = df_windows['window'].max()
        
        # Calcular os limites
        test_start = max_window - n_windows_test + 1
        val_start = test_start - n_windows_val
        
        # Dividir
        df_train = df_windows.filter(pl.col('window') < val_start)
        df_val = df_windows.filter(
            (pl.col('window') >= val_start) & (pl.col('window') < test_start)
        )
        df_test = df_windows.filter(pl.col('window') >= test_start)
        
        return df_train, df_val, df_test
    
    @staticmethod
    def _calculate_effective_period_stats(df: pl.DataFrame, target_col: str) -> dict:
        """
        Calcula estatísticas do período efetivo (primeira a última semana com atividade)
        
        Returns:
            dict com estatísticas de período efetivo e gaps
        """
        # Filtrar apenas registros com atividade (target > 0)
        df_active = df.filter(pl.col(target_col) > 0)
        
        if len(df_active) == 0:
            return {
                'has_activity': False,
                'n_vehicles_with_activity': 0,
                'effective_start': None,
                'effective_end': None,
                'effective_days': 0,
                'avg_gap': 0.0,
                'total_gaps': 0
            }
        
        # Encontrar primeira e última semana com atividade por veículo
        vehicles_stats = []
        
        for vehicle_id in df_active['veiculo_id'].unique():
            df_vehicle = df_active.filter(pl.col('veiculo_id') == vehicle_id).sort('data')
            
            if len(df_vehicle) == 0:
                continue
            
            dates = df_vehicle['data'].to_list()
            
            # Primeira e última data com atividade
            first_date = dates[0]
            last_date = dates[-1]
            
            # Ajustar para início da semana (segunda-feira)
            first_week_start = first_date - timedelta(days=first_date.weekday())
            
            # Ajustar para fim da semana (domingo)
            last_week_end = last_date + timedelta(days=(6 - last_date.weekday()))
            
            # Calcular gaps no período efetivo
            gaps = []
            for i in range(1, len(dates)):
                gap = (dates[i] - dates[i-1]).days - 1
                if gap > 0:
                    gaps.append(gap)
            
            vehicles_stats.append({
                'veiculo_id': vehicle_id,
                'effective_start': first_week_start,
                'effective_end': last_week_end,
                'effective_days': (last_week_end - first_week_start).days + 1,
                'n_active_days': len(dates),
                'gaps': gaps,
                'avg_gap': sum(gaps) / len(gaps) if gaps else 0.0,
                'max_gap': max(gaps) if gaps else 0
            })
        
        if not vehicles_stats:
            return {
                'has_activity': False,
                'n_vehicles_with_activity': 0,
                'effective_start': None,
                'effective_end': None,
                'effective_days': 0,
                'avg_gap': 0.0,
                'total_gaps': 0
            }
        
        # Agregar estatísticas
        all_gaps = [gap for v in vehicles_stats for gap in v['gaps']]
        
        return {
            'has_activity': True,
            'n_vehicles_with_activity': len(vehicles_stats),
            'effective_start': min(v['effective_start'] for v in vehicles_stats),
            'effective_end': max(v['effective_end'] for v in vehicles_stats),
            'effective_days': int(sum(v['effective_days'] for v in vehicles_stats) / len(vehicles_stats)),
            'avg_gap': sum(all_gaps) / len(all_gaps) if all_gaps else 0.0,
            'max_gap': max(all_gaps) if all_gaps else 0,
            'total_gaps': len(all_gaps)
        }
    
    def _calculate_window_gaps(
        self,
        df_windows: pl.DataFrame,
        window_type: str = 'both'
    ) -> dict:
        """
        Calcula gaps médios dentro de cada janela (history e/ou forecast)
        
        Parameters:
        -----------
        df_windows : polars.DataFrame
            DataFrame com janelas geradas
        window_type : str, optional
            Tipo de janela para calcular gaps:
            - 'history': apenas período de histórico
            - 'forecast': apenas período de forecast
            - 'both': ambos (padrão)
        
        Returns:
        --------
        dict
            Dicionário com listas de gaps médios por janela:
            - 'history_gaps': lista de gaps médios no período de histórico
            - 'forecast_gaps': lista de gaps médios no período de forecast
            - 'window_ids': lista de IDs das janelas
        """
        if len(df_windows) == 0:
            return {
                'history_gaps': [],
                'forecast_gaps': [],
                'window_ids': []
            }
        
        # Obter datas mínimas e máximas de cada janela
        window_dates = (
            df_windows
            .group_by(['veiculo_id', 'window'])
            .agg([
                pl.col('data').min().alias('min_date'),
                pl.col('data').max().alias('max_date')
            ])
        )
        
        history_gaps = []
        forecast_gaps = []
        window_ids = []
        
        for row in window_dates.iter_rows(named=True):
            vehicle_id = row['veiculo_id']
            window_id = row['window']
            min_date = row['min_date']
            max_date = row['max_date']
            
            # Calcular data de corte entre history e forecast
            # history_end = min_date + (history_size - 1) dias
            history_end = min_date + timedelta(days=self.history_size - 1)
            
            # Filtrar dados desta janela
            df_window = df_windows.filter(
                (pl.col('veiculo_id') == vehicle_id) &
                (pl.col('window') == window_id) &
                (pl.col(self.target) > 0)
            ).sort('data')
            
            if len(df_window) == 0:
                continue
            
            dates = df_window['data'].to_list()
            
            # Separar datas em history e forecast
            history_dates = [d for d in dates if d <= history_end]
            forecast_dates = [d for d in dates if d > history_end]
            
            # Calcular gaps no history
            if window_type in ['history', 'both'] and len(history_dates) > 1:
                gaps_hist = []
                for i in range(1, len(history_dates)):
                    gap = (history_dates[i] - history_dates[i-1]).days - 1
                    if gap > 0:
                        gaps_hist.append(gap)
                
                avg_gap_hist = sum(gaps_hist) / len(gaps_hist) if gaps_hist else 0.0
                history_gaps.append(avg_gap_hist)
            elif window_type in ['history', 'both']:
                history_gaps.append(0.0)
            
            # Calcular gaps no forecast
            if window_type in ['forecast', 'both'] and len(forecast_dates) > 1:
                gaps_fore = []
                for i in range(1, len(forecast_dates)):
                    gap = (forecast_dates[i] - forecast_dates[i-1]).days - 1
                    if gap > 0:
                        gaps_fore.append(gap)
                
                avg_gap_fore = sum(gaps_fore) / len(gaps_fore) if gaps_fore else 0.0
                forecast_gaps.append(avg_gap_fore)
            elif window_type in ['forecast', 'both']:
                forecast_gaps.append(0.0)
            
            window_ids.append(f"{vehicle_id}_{window_id}")
        
        return {
            'history_gaps': history_gaps,
            'forecast_gaps': forecast_gaps,
            'window_ids': window_ids
        }
    
    def plot_gap_distributions(
        self,
        result: dict,
        figsize: Tuple[int, int] = (15, 10),
        bins: int = 50,
        save_path: Optional[str] = None
    ):
        """
        Plota distribuições de gaps médios nas janelas de train/val/test
        
        Cria gráficos mostrando:
        1. Distribuição de gaps no período de histórico (history_size)
        2. Distribuição de gaps no período de forecast (forecast_horizon)
        
        Parameters:
        -----------
        result : dict
            Resultado retornado pelo método generate()
        figsize : tuple, optional
            Tamanho da figura (largura, altura) (padrão: (15, 10))
        bins : int, optional
            Número de bins para os histogramas (padrão: 50)
        save_path : str, optional
            Caminho para salvar a figura. Se None, apenas exibe (padrão: None)
        
        Examples:
        ---------
        >>> result = gen.generate(n_windows_test=4, n_windows_val=4)
        >>> gen.plot_gap_distributions(result)
        >>> gen.plot_gap_distributions(result, save_path='gap_distributions.png')
        """
        
        # Calcular gaps para cada split
        gaps_train = self._calculate_window_gaps(result['train'], window_type='both')
        gaps_val = self._calculate_window_gaps(result['val'], window_type='both')
        gaps_test = self._calculate_window_gaps(result['test'], window_type='both')
        
        # Criar figura com 2 linhas e 3 colunas
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(
            f'Distribuição de Gaps Médios por Janela\n'
            f'Target: {self.target} | History: {self.history_size}d | Forecast: {self.forecast_horizon}d',
            fontsize=14,
            fontweight='bold'
        )
        
        splits = ['train', 'val', 'test']
        gaps_data = [gaps_train, gaps_val, gaps_test]
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        
        # Linha 1: Gaps no History
        for idx, (split, gaps, color) in enumerate(zip(splits, gaps_data, colors)):
            ax = axes[0, idx]
            
            history_gaps = gaps['history_gaps']
            
            if len(history_gaps) > 0:
                ax.hist(history_gaps, bins=bins, color=color, alpha=0.7, edgecolor='black')
                
                # Estatísticas
                mean_gap = np.mean(history_gaps)
                median_gap = np.median(history_gaps)
                max_gap = np.max(history_gaps)
                
                # Linhas verticais para média e mediana
                ax.axvline(mean_gap, color='red', linestyle='--', linewidth=2, label=f'Média: {mean_gap:.2f}d')
                ax.axvline(median_gap, color='orange', linestyle='--', linewidth=2, label=f'Mediana: {median_gap:.2f}d')
                
                ax.set_title(f'{split.upper()} - History ({self.history_size}d)', fontweight='bold')
                ax.set_xlabel('Gap Médio (dias)')
                ax.set_ylabel('Frequência')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Adicionar texto com estatísticas
                stats_text = f'N janelas: {len(history_gaps)}\nMáx: {max_gap:.2f}d'
                ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            else:
                ax.text(0.5, 0.5, 'Sem dados', ha='center', va='center', fontsize=14)
                ax.set_title(f'{split.upper()} - History ({self.history_size}d)', fontweight='bold')
        
        # Linha 2: Gaps no Forecast
        for idx, (split, gaps, color) in enumerate(zip(splits, gaps_data, colors)):
            ax = axes[1, idx]
            
            forecast_gaps = gaps['forecast_gaps']
            
            if len(forecast_gaps) > 0:
                ax.hist(forecast_gaps, bins=bins, color=color, alpha=0.7, edgecolor='black')
                
                # Estatísticas
                mean_gap = np.mean(forecast_gaps)
                median_gap = np.median(forecast_gaps)
                max_gap = np.max(forecast_gaps)
                
                # Linhas verticais para média e mediana
                ax.axvline(mean_gap, color='red', linestyle='--', linewidth=2, label=f'Média: {mean_gap:.2f}d')
                ax.axvline(median_gap, color='orange', linestyle='--', linewidth=2, label=f'Mediana: {median_gap:.2f}d')
                
                ax.set_title(f'{split.upper()} - Forecast ({self.forecast_horizon}d)', fontweight='bold')
                ax.set_xlabel('Gap Médio (dias)')
                ax.set_ylabel('Frequência')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Adicionar texto com estatísticas
                stats_text = f'N janelas: {len(forecast_gaps)}\nMáx: {max_gap:.2f}d'
                ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            else:
                ax.text(0.5, 0.5, 'Sem dados', ha='center', va='center', fontsize=14)
                ax.set_title(f'{split.upper()} - Forecast ({self.forecast_horizon}d)', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Gráfico salvo em: {save_path}")
        
        plt.show()
    
    def generate(
        self, 
        n_windows_test: int, 
        n_windows_val: int, 
        skip_n_days: int = 0,
        use_effective_period: bool = False,
        output_dir: Optional[str] = None,
        suffix: Optional[Union[str, int]] = None,
        verbose: bool = True
    ):
        """
        Executa todo o fluxo de geração de datasets de treino, validação e teste
        
        Este método:
        1. Remove os últimos skip_n_days de cada veículo (se especificado)
        2. Gera TODAS as janelas do dataset completo com stride configurado
        3. Divide as janelas em treino/validação/teste
        4. Salva os datasets (se output_dir especificado)
        
        IMPORTANTE: A divisão train/val/test é feita APENAS nas janelas geradas,
        garantindo consistência independente da configuração. Pode haver overlap
        temporal entre os splits.
        
        Parameters:
        -----------
        n_windows_test : int
            Número de janelas para o conjunto de teste
        n_windows_val : int
            Número de janelas para o conjunto de validação
        skip_n_days : int, optional
            Número de dias a remover do final da série temporal de cada veículo (padrão: 0)
        use_effective_period: bool,
            Se true, ignora o max_gap do dataset de entrada e usa o período efetivo como
            segmento na geração de janelas (padrão: False)
        output_dir : str, optional
            Diretório para salvar os datasets. Se None, não salva (padrão: None)
        suffix : int, str, optional
            Sufixo para diferenciar diferentes datasets (padrão: None)
        verbose : bool, optional
            Se True, imprime informações sobre os datasets gerados (padrão: True)
        
        Returns:
        --------
        dict
            Dicionário contendo os datasets e metadados
        """
        
        if verbose:
            print("\n" + "="*80)
            print("GERAÇÃO DE DATASETS")
            print("="*80)
            print(f"\nConfiguração:")
            print(f"  Target: {self.target}")
            print(f"  History: {self.history_size}d")
            print(f"  Forecast: {self.forecast_horizon}d")
            print(f"  Window size: {self.history_size + self.forecast_horizon}d")
            print(f"  Stride: {self.stride}d")
            print(f"  Max gap: {self.max_gap}d (ignorado)" if use_effective_period else f"  Max gap: {self.max_gap}d")
            print(f"  Windows: {n_windows_val}(val) + {n_windows_test}(test)")
            if skip_n_days > 0:
                print(f"  Skip: {skip_n_days}d")
            print()
        
        # ========================================================================
        # CONTADORES INICIAIS
        # ========================================================================
        n_vehicles_initial = self.df['veiculo_id'].n_unique()
        
        # Veículos com atividade inicial (target > 0)
        n_vehicles_with_activity_initial = self.df.filter(
            pl.col(self.target) > 0
        )['veiculo_id'].n_unique()
        
        # ========================================================================
        # PREPARAR DADOS (SKIP)
        # ========================================================================
        df_prepared = self._prepare_data(skip_n_days=skip_n_days)
        
        # Veículos com atividade após skip
        n_vehicles_with_activity_after_skip = df_prepared.filter(
            pl.col(self.target) > 0
        )['veiculo_id'].n_unique()
        
        # ========================================================================
        # GERAR TODAS AS JANELAS DO DATASET COMPLETO
        # ========================================================================
        if verbose:
            print("\n" + "="*80)
            print("GERANDO JANELAS DO DATASET COMPLETO")
            print("="*80)
        
        windows_builder = DatasetSlicer(
            target=self.target, 
            window_size=self.history_size + self.forecast_horizon,
            stride=self.stride,
            max_gap=self.max_gap if not use_effective_period else None
        )
                
        df_windows_all, df_periods_all = windows_builder.slice(
            df_prepared, 
            verbose=verbose
        )
        
        if len(df_windows_all) == 0:
            if verbose:
                print("\n⚠️  AVISO: Nenhuma janela foi gerada!")
                print("="*80 + "\n")
            
            # Retornar datasets vazios
            empty_df = pl.DataFrame(schema={
                'veiculo_id': pl.Int64,
                'window': pl.Int64,
                'data': pl.Date,
                self.target: pl.Float64
            })
            
            return {
                "train": empty_df,
                "val": empty_df,
                "test": empty_df,
                "train_metadata": df_periods_all,
                "val_metadata": df_periods_all,
                "test_metadata": df_periods_all,
                "stats": {}
            }
        
        # ========================================================================
        # DIVIDIR JANELAS EM TRAIN/VAL/TEST
        # ========================================================================
        if verbose:
            print("\n" + "="*80)
            print("DIVIDINDO JANELAS EM TRAIN/VAL/TEST")
            print("="*80)
            print(f"\nTotal de janelas geradas: {df_windows_all['window'].n_unique()}")
        
        df_windows_train, df_windows_val, df_windows_test = self._split_windows(
            df_windows_all,
            n_windows_test=n_windows_test,
            n_windows_val=n_windows_val
        )
        
        if verbose:
            print(f"  Train: {df_windows_train['window'].n_unique() if len(df_windows_train) > 0 else 0} janelas")
            print(f"  Val:   {df_windows_val['window'].n_unique() if len(df_windows_val) > 0 else 0} janelas")
            print(f"  Test:  {df_windows_test['window'].n_unique() if len(df_windows_test) > 0 else 0} janelas")
        
        # Filtrar metadados para cada split (baseado nos veículos presentes)
        vehicles_train = set(df_windows_train['veiculo_id'].unique()) if len(df_windows_train) > 0 else set()
        vehicles_val = set(df_windows_val['veiculo_id'].unique()) if len(df_windows_val) > 0 else set()
        vehicles_test = set(df_windows_test['veiculo_id'].unique()) if len(df_windows_test) > 0 else set()
        
        df_periods_train = df_periods_all.filter(pl.col('veiculo_id').is_in(list(vehicles_train)))
        df_periods_val = df_periods_all.filter(pl.col('veiculo_id').is_in(list(vehicles_val)))
        df_periods_test = df_periods_all.filter(pl.col('veiculo_id').is_in(list(vehicles_test)))
        
        # ========================================================================
        # CALCULAR ESTATÍSTICAS
        # ========================================================================
        # Contar veículos finais (com atividade nas janelas)
        n_vehicles_train = df_windows_train.filter(
            pl.col(self.target) > 0
        )['veiculo_id'].n_unique() if len(df_windows_train) > 0 else 0
        
        n_vehicles_val = df_windows_val.filter(
            pl.col(self.target) > 0
        )['veiculo_id'].n_unique() if len(df_windows_val) > 0 else 0
        
        n_vehicles_test = df_windows_test.filter(
            pl.col(self.target) > 0
        )['veiculo_id'].n_unique() if len(df_windows_test) > 0 else 0
        
        vehicles_final = set()
        if n_vehicles_train > 0:
            vehicles_final.update(
                df_windows_train.filter(pl.col(self.target) > 0)['veiculo_id'].unique()
            )
        if n_vehicles_val > 0:
            vehicles_final.update(
                df_windows_val.filter(pl.col(self.target) > 0)['veiculo_id'].unique()
            )
        if n_vehicles_test > 0:
            vehicles_final.update(
                df_windows_test.filter(pl.col(self.target) > 0)['veiculo_id'].unique()
            )
        
        n_vehicles_final = len(vehicles_final)
        
        # Estatísticas de período efetivo
        stats_windows_train = self._calculate_effective_period_stats(df_windows_train, self.target)
        stats_windows_val = self._calculate_effective_period_stats(df_windows_val, self.target)
        stats_windows_test = self._calculate_effective_period_stats(df_windows_test, self.target)
        
        # ========================================================================
        # PRINTS
        # ========================================================================
        if verbose:
            # Tabela resumo de veículos
            print("\n" + "="*80)
            print("RESUMO - VEÍCULOS COM ATIVIDADE POR ETAPA")
            print("-"*90)
            print(f"{'Etapa':<35} {'Veículos':>12} {'Perda':>12} {'% Perda':>10} {'Gap Médio':>12}")
            print("-"*90)
            
            # Inicial
            print(f"{'Dataset inicial':<35} {n_vehicles_with_activity_initial:>12,} {'-':>12} {'-':>10} {'-':>12}")
            
            # Após skip
            loss_skip = n_vehicles_with_activity_initial - n_vehicles_with_activity_after_skip
            pct_skip = (loss_skip / n_vehicles_with_activity_initial * 100) if n_vehicles_with_activity_initial > 0 else 0
            
            # Gap médio após skip
            stats_after_skip = self._calculate_effective_period_stats(df_prepared, self.target)
            avg_gap_after_skip = stats_after_skip['avg_gap'] if stats_after_skip['has_activity'] else 0.0
            
            print(f"{'Após skip temporal':<35} {n_vehicles_with_activity_after_skip:>12,} {loss_skip:>12,} {pct_skip:>9.1f}% {avg_gap_after_skip:>11.1f}d")
            
            # Após geração de janelas
            loss_windows = n_vehicles_with_activity_after_skip - n_vehicles_final
            pct_windows = (loss_windows / n_vehicles_with_activity_after_skip * 100) if n_vehicles_with_activity_after_skip > 0 else 0
            
            # Gap médio nas janelas geradas
            all_gaps_windows = []
            for stats in [stats_windows_train, stats_windows_val, stats_windows_test]:
                if stats['has_activity'] and stats['total_gaps'] > 0:
                    all_gaps_windows.append(stats['avg_gap'])
            avg_gap_windows = sum(all_gaps_windows) / len(all_gaps_windows) if all_gaps_windows else 0.0
            
            print(f"{'Após geração de janelas':<35} {n_vehicles_final:>12,} {loss_windows:>12,} {pct_windows:>9.1f}% {avg_gap_windows:>11.1f}d")
            
            print("-"*90)
            
            # Perda total
            loss_total = n_vehicles_with_activity_initial - n_vehicles_final
            pct_total = (loss_total / n_vehicles_with_activity_initial * 100) if n_vehicles_with_activity_initial > 0 else 0
            print(f"{'PERDA TOTAL':<35} {n_vehicles_final:>12,} {loss_total:>12,} {pct_total:>9.1f}% {avg_gap_windows:>11.1f}d")
            print("-"*90)
            print()
            
            # Tabela de datasets gerados
            print("DATASETS GERADOS - PERÍODO EFETIVO")
            print("-"*105)
            print(f"{'Split':<10} {'Período Efetivo':<30} {'Veículos':>10} {'Janelas':>10} {'Registros':>12} {'Gap Médio':>12}")
            print("-"*105)
            
            # Treino
            if stats_windows_train['has_activity']:
                period_train = f"{stats_windows_train['effective_start']} - {stats_windows_train['effective_end']}"
                gap_train = stats_windows_train['avg_gap']
            else:
                period_train = "Sem atividade"
                gap_train = 0.0
            
            print(f"{'Treino':<10} {period_train:<30} {n_vehicles_train:>10,} "
                  f"{df_windows_train['window'].n_unique() if len(df_windows_train) > 0 else 0:>10} "
                  f"{len(df_windows_train):>12,} {gap_train:>11.1f}d")
            
            # Validação
            if stats_windows_val['has_activity']:
                period_val = f"{stats_windows_val['effective_start']} - {stats_windows_val['effective_end']}"
                gap_val = stats_windows_val['avg_gap']
            else:
                period_val = "Sem atividade"
                gap_val = 0.0
            
            print(f"{'Validação':<10} {period_val:<30} {n_vehicles_val:>10,} "
                  f"{df_windows_val['window'].n_unique() if len(df_windows_val) > 0 else 0:>10} "
                  f"{len(df_windows_val):>12,} {gap_val:>11.1f}d")
            
            # Teste
            if stats_windows_test['has_activity']:
                period_test = f"{stats_windows_test['effective_start']} - {stats_windows_test['effective_end']}"
                gap_test = stats_windows_test['avg_gap']
            else:
                period_test = "Sem atividade"
                gap_test = 0.0
            
            print(f"{'Teste':<10} {period_test:<30} {n_vehicles_test:>10,} "
                  f"{df_windows_test['window'].n_unique() if len(df_windows_test) > 0 else 0:>10} "
                  f"{len(df_windows_test):>12,} {gap_test:>11.1f}d")
            
            print("-"*105)
            print()
        
        # ========================================================================
        # PREPARAR RESULTADO
        # ========================================================================
        result = {
            "train": df_windows_train, 
            "val": df_windows_val, 
            "test": df_windows_test, 
            "train_metadata": df_periods_train, 
            "val_metadata": df_periods_val, 
            "test_metadata": df_periods_test
        }
        
        # Estatísticas detalhadas
        result['stats'] = {
            'n_vehicles_initial': n_vehicles_initial,
            'n_vehicles_with_activity_initial': n_vehicles_with_activity_initial,
            'n_vehicles_with_activity_after_skip': n_vehicles_with_activity_after_skip,
            'n_vehicles_final': n_vehicles_final,
            'loss_skip': loss_skip,
            'loss_windows': loss_windows,
            'loss_total': loss_total,
            'pct_skip': pct_skip,
            'pct_windows': pct_windows,
            'pct_total': pct_total,
            'avg_gap_after_skip': avg_gap_after_skip,
            'avg_gap_windows': avg_gap_windows,
            'train': {
                'n_vehicles': n_vehicles_train,
                'effective_period': stats_windows_train,
                'avg_gap': gap_train if stats_windows_train['has_activity'] else 0.0
            },
            'val': {
                'n_vehicles': n_vehicles_val,
                'effective_period': stats_windows_val,
                'avg_gap': gap_val if stats_windows_val['has_activity'] else 0.0
            },
            'test': {
                'n_vehicles': n_vehicles_test,
                'effective_period': stats_windows_test,
                'avg_gap': gap_test if stats_windows_test['has_activity'] else 0.0
            }
        }
        
        # ========================================================================
        # SALVAR DATASETS
        # ========================================================================
        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Construir nome do diretório
            dirname_parts = [
                self._dataset_name,
                f'hist{self.history_size}',
                f'fh{self.forecast_horizon}',
                f'stride{self.stride}',
                f'ntest{n_windows_test}',
                f'nval{n_windows_val}'
            ]
            
            if use_effective_period:
                dirname_parts.append(f'useeffectiveperiod')
            
            if skip_n_days > 0:
                dirname_parts.append(f'skip{skip_n_days}')
            
            if suffix is not None:
                dirname_parts.append(f'ds{suffix}')
            
            dirname = '_'.join(dirname_parts)
            dataset_dir = output_path / dirname
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            # Salvar arquivos
            for split in ['train', 'val', 'test']:
                result[split].write_csv(dataset_dir / f'{split}.csv')
                result[f'{split}_metadata'].write_csv(dataset_dir / f'{split}_metadata.csv')
            
            # Salvar estatísticas
            with open(dataset_dir / 'stats.json', 'w') as f:
                # Converter objetos date para string
                stats_to_save = result['stats'].copy()
                for split in ['train', 'val', 'test']:
                    if stats_to_save[split]['effective_period']['effective_start']:
                        stats_to_save[split]['effective_period']['effective_start'] = \
                            stats_to_save[split]['effective_period']['effective_start'].isoformat()
                    if stats_to_save[split]['effective_period']['effective_end']:
                        stats_to_save[split]['effective_period']['effective_end'] = \
                            stats_to_save[split]['effective_period']['effective_end'].isoformat()
                
                json.dump(stats_to_save, f, indent=2)
            
            result['output_path'] = dataset_dir
            
            if verbose:
                print(f"✓ Datasets salvos em: {dataset_dir.name}\n")
        
        if verbose:
            print("="*80)
            print("GERAÇÃO CONCLUÍDA")
            print("="*80 + "\n")
        
        return result
    
    def __repr__(self) -> str:
        """Representação em string do objeto"""
        return (
            f"DatasetGenerator(\n"
            f"  target='{self.target}',\n"
            f"  history_size={self.history_size},\n"
            f"  forecast_horizon={self.forecast_horizon},\n"
            f"  stride={self.stride},\n"
            f"  max_gap={self.max_gap},\n"
            f"  n_vehicles={self.df['veiculo_id'].n_unique()},\n"
            f"  n_records={len(self.df):,}\n"
            f")"
        )
    
def merge_data(
    path_raw_data: str,
    *path_data: str,
    output_dir: Optional[str] = None
) -> pl.DataFrame:
    """
    Combina múltiplos datasets com targets diferentes
    
    Parameters:
    -----------
    path_raw_data : str
        Caminho do dataset inicial completo
    *path_data : str
        Caminhos dos datasets processados
        
    Returns:
    --------
    pl.DataFrame
        DataFrame combinado
    """
    
    print("=" * 80)
    print("MERGE DATA")
    print("=" * 80)
    print()
    
    if len(path_data) == 0:
        raise ValueError("Pelo menos um path_data deve ser fornecido")
    
    # Extrair targets e características
    targets = []
    characteristics = None
    
    for path in path_data:
        filename = Path(path).stem
        
        match_target = re.search(r'dataset_([^_]+(?:_[^_]+)*)_mindays', filename)
        if not match_target:
            raise ValueError(f"Padrão inválido: {filename}")
        
        target = match_target.group(1)
        targets.append(target)
        
        match_chars = re.search(r'(mindays\d+_minweeks\d+_maxgap\d+)', filename)
        if not match_chars:
            raise ValueError(f"Características não encontradas: {filename}")
        
        chars = match_chars.group(1)
        
        if characteristics is None:
            characteristics = chars
        elif characteristics != chars:
            raise ValueError(f"Características inconsistentes: {chars} != {characteristics}")
    
    print(f"Targets: {', '.join(targets)}")
    print(f"Características: {characteristics}")
    print()
    
    # Ler datasets processados (apenas veiculo_id e data)
    print("Lendo datasets processados...")
    
    dfs = []
    for path in path_data:
        df = pl.read_csv(path, columns=['veiculo_id', 'data'], try_parse_dates=True)
        dfs.append(df)
        print(f"  • {Path(path).name}: {len(df):,} registros")
    
    # Concatenar todos os dataframes
    print("\nConcatenando chaves...")
    df_keys = pl.concat(dfs)
    
    print(f"Total após concatenação: {len(df_keys):,}")
    
    # Remover duplicatas (veiculo_id, data) e ordenar
    df_keys = df_keys.unique(subset=['veiculo_id', 'data']).sort(['veiculo_id', 'data'])
    
    print(f"Chaves únicas: {len(df_keys):,}")
    print(f"Veículos únicos: {df_keys['veiculo_id'].n_unique()}")
    print()
    
    # Ler raw
    print("Lendo dataset raw...")
    
    cols_raw = ['veiculo_id', 'data'] + targets
    df_raw = pl.read_csv(path_raw_data, columns=cols_raw, try_parse_dates=True)
    
    # Converter data se necessário
    if df_raw['data'].dtype not in [pl.Date, pl.Datetime]:
        df_raw = df_raw.with_columns(
            pl.col('data').str.strptime(pl.Date, '%d/%m/%Y')
        )
    
    print(f"  • Registros: {len(df_raw):,}")
    print(f"  • Veículos: {df_raw['veiculo_id'].n_unique()}")
    print()
    
    # Join com raw
    print("Fazendo join com raw...")
    
    df_merged = df_keys.join(
        df_raw,
        on=['veiculo_id', 'data'],
        how='left'
    )
    
    # Verificar nulos
    print("\nVerificando valores nulos:")
    has_nulls = False
    
    for target in targets:
        n_nulls = df_merged[target].null_count()
        if n_nulls > 0:
            has_nulls = True
            pct_null = n_nulls / len(df_merged) * 100
            print(f"  ⚠️  {target}: {n_nulls:,} nulos ({pct_null:.2f}%)")
        else:
            print(f"  ✓ {target}: sem nulos")
    
    if has_nulls:
        print("\n⚠️  ERRO: Valores nulos encontrados após join!")
        print("Existem pares (veiculo_id, data) nos datasets processados")
        print("que não existem no dataset raw.")
        print()
        
        # Mostrar exemplos
        print("Exemplos de registros com nulos:")
        null_mask = pl.any_horizontal([pl.col(t).is_null() for t in targets])
        print(df_merged.filter(null_mask).head(10))
        print()
        
        raise ValueError("Join com raw resultou em valores nulos. Verifique os dados de entrada.")
    
    print()
    print(f"Dataset final:")
    print(f"  • Registros: {len(df_merged):,}")
    print(f"  • Veículos: {df_merged['veiculo_id'].n_unique()}")
    print(f"  • Período: {df_merged['data'].min()} a {df_merged['data'].max()}")
    print(f"  • Colunas: {', '.join(df_merged.columns)}")
    print()
    
    # Salvar
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        targets_str = '_'.join(sorted(targets))
        filename = f"dataset_{targets_str}_{characteristics}.csv"
        filepath = output_path / filename
        
        print(f"Salvando: {filepath}")
        df_merged.write_csv(filepath)
        
        file_size = filepath.stat().st_size / (1024 * 1024)
        print(f"Tamanho: {file_size:.2f} MB")
        print()
    
    print("=" * 80)
    print("CONCLUÍDO")
    print("=" * 80)
    print()
    
    return df_merged


class ClassSplitter:
    """
    Divisor de dados por classes/segmentos
    
    Esta classe divide um dataset em múltiplos arquivos baseado em
    classificações ou segmentações de veículos, facilitando análises
    e treinamentos específicos por grupo.
    
    Attributes
    ----------
    df_classification : pd.DataFrame or pl.DataFrame
        DataFrame contendo a classificação dos veículos com colunas:
        - veiculo_id: ID do veículo
        - cluster_segmento: Classe/cluster do veículo
    class_column : str
        Nome da coluna que contém a classificação (padrão: 'cluster_segmento')
    vehicle_id_column : str
        Nome da coluna de ID do veículo (padrão: 'veiculo_id')
    
    Examples
    --------
    Uso básico com pandas:
    
    >>> df_class = pd.read_csv('veiculos_segmentados.csv')
    >>> df_class = df_class[['cluster_segmento', 'veiculo_id']]
    >>> 
    >>> splitter = ClassSplitter(df_class)
    >>> splitter.split(
    ...     data_path='dataset_km_dia_clean.csv',
    ...     output_dir='./segments'
    ... )
    Dividindo dataset_km_dia_clean.csv em 3 classes...
      ✓ Classe 0: 1,250 veículos, 125,430 registros
      ✓ Classe 1: 980 veículos, 98,230 registros
      ✓ Classe 2: 450 veículos, 45,120 registros
    ✓ 3 arquivos salvos em: ./segments
    
    Filtrar por tipo de veículo antes de dividir:
    
    >>> df_all = pd.read_csv('veiculos_segmentados_final.csv')
    >>> df_deslocamento = df_all[
    ...     ~df_all['is_outlier'] & 
    ...     (df_all['tipo_veiculo'] == 'Deslocamento')
    ... ][['cluster_segmento', 'veiculo_id']]
    >>> 
    >>> splitter = ClassSplitter(df_deslocamento)
    >>> splitter.split('dataset_km_dia_clean.csv')
    """
    
    def __init__(
        self,
        df_classification: pd.DataFrame,
        class_column: str = 'cluster_segmento',
        vehicle_id_column: str = 'veiculo_id'
    ):
        """
        Inicializa o divisor de classes
        Parameters
        ----------
        df_classification : pd.DataFrame
            DataFrame contendo a classificação dos veículos.
            Deve conter no mínimo as colunas de ID do veículo e classe
        class_column : str, default='cluster_segmento'
            Nome da coluna que contém a classificação/cluster
        vehicle_id_column : str, default='veiculo_id'
            Nome da coluna de ID do veículo
            
        Raises
        ------
        ValueError
            Se as colunas especificadas não existirem no DataFrame
            
        Examples
        --------
        >>> df_class = pd.DataFrame({
        ...     'veiculo_id': [1, 2, 3, 4],
        ...     'cluster_segmento': [0, 0, 1, 1]
        ... })
        >>> splitter = ClassSplitter(df_class)
        
        Com nomes de colunas customizados:
        
        >>> df_class = pd.DataFrame({
        ...     'vehicle_id': [1, 2, 3],
        ...     'segment': ['A', 'A', 'B']
        ... })
        >>> splitter = ClassSplitter(
        ...     df_class,
        ...     class_column='segment',
        ...     vehicle_id_column='vehicle_id'
        ... )
        """
        # Validar colunas
        if class_column not in df_classification.columns:
            raise ValueError(
                f"Coluna '{class_column}' não encontrada no DataFrame. "
                f"Colunas disponíveis: {list(df_classification.columns)}"
            )
        
        if vehicle_id_column not in df_classification.columns:
            raise ValueError(
                f"Coluna '{vehicle_id_column}' não encontrada no DataFrame. "
                f"Colunas disponíveis: {list(df_classification.columns)}"
            )
        
        self.df_classification = df_classification
        self.class_column = class_column
        self.vehicle_id_column = vehicle_id_column
        
        # Informações sobre as classes
        self.classes = sorted(df_classification[class_column].unique())
        self.n_classes = len(self.classes)
        self.n_vehicles = len(df_classification)
    
    def split(
        self,
        data_path: str,
        output_dir: Optional[str] = None,
        suffix: str = 'class',
        save_index: bool = False,
        verbose: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Divide dados em múltiplos arquivos por classe
        Parameters
        ----------
        data_path : str
            Caminho do arquivo de dados a ser dividido.
            Deve conter uma coluna com ID do veículo
        output_dir : str, optional
            Diretório de saída para os arquivos divididos.
            Se None, cria subdiretório 'segments' no diretório do arquivo original
        suffix : str, default='class'
            Sufixo para nomear os arquivos divididos.
            Formato: {nome_original}_{suffix},{classe}.csv
        save_index : bool, default=False
            Se True, salva o índice do DataFrame nos arquivos CSV
        verbose : bool, default=True
            Se True, imprime informações sobre o processo
      Returns
        -------
        dict
            Dicionário com DataFrames divididos por classe:
            {classe: DataFrame}
            
        Raises
        ------
        FileNotFoundError
            Se o arquivo data_path não existir
        ValueError
            Se a coluna de vehicle_id não existir nos dados
            
        Examples
        --------
        Uso básico:
        
        >>> splitter = ClassSplitter(df_class)
        >>> result = splitter.split('dataset_km_dia_clean.csv')
        >>> print(result.keys())
        dict_keys([0, 1, 2])
        
        Especificar diretório de saída:
        
        >>> splitter.split(
        ...     data_path='dataset_km_dia_clean.csv',
        ...     output_dir='./output/segments',
        ...     suffix='cluster'
        ... )
        
        Sem salvar arquivos (apenas retornar dicionário):
        
        >>> result = splitter.split(
        ...     data_path='dataset_km_dia_clean.csv',
        ...     output_dir=None,
        ...     verbose=False
        ... )
        
        Salvar múltiplos datasets:
        
        >>> datasets = [
        ...     'dataset_km_dia_clean_mindays7.csv',
        ...     'dataset_km_dia_clean_mindays10.csv'
        ... ]
        >>> for dataset in datasets:
        ...     splitter.split(dataset)
        
        Notes
        -----
        - Apenas veículos presentes em df_classification serão incluídos
        - A ordem das colunas do arquivo original é preservada
        - Classes sem veículos são ignoradas
        """
        # Validar arquivo
        file_path = Path(data_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {data_path}")
        
        # Carregar dados
        df_data = pd.read_csv(file_path)
        data_name = file_path.stem
        
        # Validar coluna de veículo
        if self.vehicle_id_column not in df_data.columns:
            raise ValueError(
                f"Coluna '{self.vehicle_id_column}' não encontrada nos dados. "
                f"Colunas disponíveis: {list(df_data.columns)}"
            )
        
        if verbose:
            print(f"\nDividindo {file_path.name} em {self.n_classes} classes...")
        
        # Merge dos dados com classificação
        df_merged = pd.merge(
            left=self.df_classification,
            right=df_data,
            how='inner',
            on=self.vehicle_id_column
        )
        
        # Determinar diretório de saída
        if output_dir is None:
            output_path = file_path.parent / 'segments'
        else:
            output_path = Path(output_dir)
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Dividir por classe
        result = {}
        saved_files = []
        
        for cls in self.classes:
            # Filtrar dados da classe
            df_class = df_merged[df_merged[self.class_column] == cls]
            
            # Manter apenas colunas originais
            df_class = df_class[df_data.columns]
            
            # Armazenar no resultado
            result[cls] = df_class
            
            # Salvar arquivo
            output_file = output_path / f"{data_name}_{suffix}{cls}.csv"
            df_class.to_csv(output_file, index=save_index)
            saved_files.append(output_file)

            if verbose:
                n_vehicles = df_class[self.vehicle_id_column].nunique()
                n_records = len(df_class)
                print(f"  ✓ Classe {cls}: {n_vehicles:,} veículos, {n_records:,} registros")

        output_metadata_file = output_path / f"metadata_{data_name}.csv"
        self.df_classification.to_csv(output_metadata_file, index=False)
        
        if verbose:
            print(f"\n✓ {len(saved_files)} arquivos salvos em: {output_path}")
            print(f"\n✓ Arquivo de metadados salvo em: {output_metadata_file}")
            print()
        
        return result
    
    def get_class_info(self) -> pd.DataFrame:
        """
        Retorna informações sobre as classes
        Returns
        -------
        pd.DataFrame
            DataFrame com informações das classes:
            - classe: Identificador da classe
            - n_veiculos: Número de veículos na classe
            - pct_veiculos: Percentual de veículos na classe
            
        Examples
        --------
        >>> splitter = ClassSplitter(df_class)
        >>> info = splitter.get_class_info()
        >>> print(info)
           classe  n_veiculos  pct_veiculos
        0       0        1250          45.5
        1       1         980          35.6
        2       2         520          18.9
        """
        info = []
        
        for cls in self.classes:
            n_vehicles = (self.df_classification[self.class_column] == cls).sum()
            pct = (n_vehicles / self.n_vehicles) * 100
            
            info.append({
                'classe': cls,
                'n_veiculos': n_vehicles,
                'pct_veiculos': pct
            })
        
        return pd.DataFrame(info)
    
    def __repr__(self) -> str:
        """
        Representação em string do objeto
        
        Returns
        -------
        str
            Representação formatada do ClassSplitter
            
        Examples
        --------
        >>> splitter = ClassSplitter(df_class)
        >>> print(splitter)
        ClassSplitter(
          n_classes=3,
          n_vehicles=2750,
          classes=[0, 1, 2],
          class_column='cluster_segmento',
          vehicle_id_column='veiculo_id'
        )
        """
        return (
            f"ClassSplitter(\n"
            f"  n_classes={self.n_classes},\n"
            f"  n_vehicles={self.n_vehicles},\n"
            f"  classes={self.classes},\n"
            f"  class_column='{self.class_column}',\n"
            f"  vehicle_id_column='{self.vehicle_id_column}'\n"
            f")"
        )


def to_local_dataset(
    datasets_dir: str,
    raw_dataset_path: str,
    history_size: int = 28,
    merge_windows: bool = False,
    output_dir: str = '../datasets/local',
    verbose: bool = True
):
    datasets_dir = Path(datasets_dir)
    output_dir = Path(output_dir)
    if verbose:
        print("=" * 80)
        print("🔄 CRIANDO DATASETS LOCAIS")
        print("=" * 80)
        print(f"Datasets dir: {datasets_dir}")
        print(f"Raw data: {Path(raw_dataset_path).name}")
        print(f"History size: {history_size} dias")
        print(f"Merge windows: {merge_windows}")
        print(f"Output dir: {output_dir}")
        print()
    # Carregar dataset bruto completo
    if verbose:
        print(f"📂 Carregando dataset bruto...")
    df_raw = (
        pl.read_csv(raw_dataset_path)
        .with_columns(
            pl.col('data').str.strptime(pl.Date, '%d/%m/%Y')
        )
    )
    if verbose:
        print(f"  ✓ {len(df_raw):,} registros carregados")
        print()
    # Descobrir datasets
    dataset_dirs = [d for d in datasets_dir.iterdir() if d.is_dir() and d.name.startswith('dataset_')]
    if not dataset_dirs:
        raise ValueError(f"Nenhum dataset encontrado em {datasets_dir}")
    if verbose:
        print(f"✓ {len(dataset_dirs)} dataset(s) encontrado(s)")
        print()
    # Processar cada dataset
    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        if verbose:
            print("=" * 80)
            print(f"📊 PROCESSANDO: {dataset_name}")
            print("=" * 80)
        # Extrair nome base (até o primeiro _mindays ou _class)
        match_base = re.match(r'(dataset_[^_]+(?:_[^_]+)*?)(_class\d+)?_mindays', dataset_name)
        if not match_base:
            warnings.warn(f"Não foi possível extrair nome base de {dataset_name}. Pulando.")
            continue
        base_name = match_base.group(1)
        # Extrair classe (ex: _class0)
        match_class = re.search(r'_class(\d+)', dataset_name)
        class_id = match_class.group(1) if match_class else '0'
        class_file = f'class{class_id}.csv'
        # Extrair forecast_horizon
        match_fh = re.search(r'_fh(\d+)_', dataset_name)
        if not match_fh:
            warnings.warn(f"Não foi possível extrair forecast_horizon de {dataset_name}. Pulando.")
            continue
        forecast_horizon = int(match_fh.group(1))
        if verbose:
            print(f"Base name: {base_name}")
            print(f"Classe: {class_id}")
            print(f"Forecast horizon: {forecast_horizon} dias")
            print()
        # Extrair target
        match_target = re.search(r'dataset_([^_]+(?:_[^_]+)*)_mindays', dataset_name)
        if not match_target:
            warnings.warn(f"Não foi possível extrair target de {dataset_name}. Pulando.")
            continue
        target = match_target.group(1)
        # Verificar se target existe no dataset bruto
        if target not in df_raw.columns:
            warnings.warn(f"Coluna '{target}' não encontrada no dataset bruto. Pulando {dataset_name}.")
            continue
        # Carregar arquivo de teste
        test_file = dataset_dir / 'test.csv'
        if not test_file.exists():
            warnings.warn(f"Arquivo test.csv não encontrado em {dataset_name}. Pulando.")
            continue
        df_test = (
            pl.read_csv(test_file)
            .with_columns(pl.col('data').str.strptime(pl.Date, '%Y-%m-%d'))
        )
        if verbose:
            print(f"✓ Test.csv carregado: {len(df_test):,} registros")
        # Extrair relação veículo × janela
        df_window_vehicles = df_test.select(['veiculo_id', 'window']).unique()
        # Construir limites temporais das janelas
        window_size = forecast_horizon + history_size
        df_window_original = (
            df_test
            .group_by('window')
            .agg(pl.col('data').max().alias('dt_fim'))
            .with_columns(
                (pl.col('dt_fim') - pl.duration(days=window_size - 1)).alias('dt_inicio')
            )
        )
        # Agrupar janelas se necessário
        if merge_windows:
            df_window_groups = _merge_sequential_windows(
                df_window_vehicles, 
                df_window_original,
                verbose
            )
        else:
            df_window_groups = (
                df_window_vehicles
                .with_columns(pl.col('window').cast(pl.Utf8).alias('window_group'))
                .join(df_window_original, on='window', how='inner')
                .select(['veiculo_id', 'window_group', 'dt_inicio', 'dt_fim'])
            )
        # Filtrar dataset bruto para este target
        df_raw_target = df_raw.select(['veiculo_id', 'data', target])
        # Merge com janelas
        df_tmp = df_raw_target.join(df_window_groups, on='veiculo_id', how='inner')
        # Filtrar datas dentro do intervalo
        df_segments = (
            df_tmp
            .filter(
                (pl.col('data') >= pl.col('dt_inicio')) &
                (pl.col('data') <= pl.col('dt_fim'))
            )
            .select(['veiculo_id', 'window_group', 'data', target])
            .sort(['veiculo_id', 'window_group', 'data'])
        )
        # Converter para pandas
        df_pd = df_segments.to_pandas()
        df_pd = df_pd.rename(columns={'window_group': 'window'})
        # Separar por tamanho de forecast
        datasets_by_forecast_size = {}
        for vehicle_id in sorted(df_pd['veiculo_id'].unique()):
            mask_vehicle = df_pd['veiculo_id'] == vehicle_id
            vehicle_data = df_pd[mask_vehicle].copy()
            for window_group in sorted(vehicle_data['window'].unique()):
                mask_window = vehicle_data['window'] == window_group
                window_data = vehicle_data[mask_window].copy()
                # Calcular forecast size
                if isinstance(window_group, str) and '-' in window_group:
                    n_windows = len(window_group.split('-'))
                else:
                    n_windows = 1
                forecast_size = forecast_horizon * n_windows
                total_size = history_size + forecast_size
                # Verificar tamanho
                if len(window_data) < total_size:
                    warnings.warn(
                        f"Veículo {vehicle_id}, janela {window_group}: "
                        f"dados insuficientes ({len(window_data)} < {total_size}). Pulando."
                    )
                    continue
                # Pegar apenas o necessário
                window_data = window_data.iloc[:total_size]
                # Adicionar ao dataset correspondente
                if forecast_size not in datasets_by_forecast_size:
                    datasets_by_forecast_size[forecast_size] = []
                datasets_by_forecast_size[forecast_size].append(window_data)
        # Salvar datasets
        if merge_windows:
            # Para cada forecast_size, salvar em base_name_hist{history_size}_fh{forecast_size}/class{class_id}.csv
            for forecast_size, df_list in sorted(datasets_by_forecast_size.items()):
                out_dir = output_dir / f"{base_name}_hist{history_size}_fh{forecast_size}"
                out_dir.mkdir(parents=True, exist_ok=True)
                # Concatenar todos os grupos em um único arquivo por classe
                df_concat = pd.concat(df_list, ignore_index=True)
                output_path = out_dir / class_file
                df_concat.to_csv(output_path, index=False)
                if verbose:
                    print(f"  ✓ {output_path}: {len(df_concat):,} registros, {df_concat['veiculo_id'].nunique()} veículos")
        else:
            # Salvar em base_name_hist{history_size}_fh{forecast_horizon}/class{class_id}.csv
            for forecast_size, df_list in datasets_by_forecast_size.items():
                out_dir = output_dir / f"{base_name}_hist{history_size}_fh{forecast_horizon}"
                out_dir.mkdir(parents=True, exist_ok=True)
                df_concat = pd.concat(df_list, ignore_index=True)
                output_path = out_dir / class_file
                df_concat.to_csv(output_path, index=False)
                if verbose:
                    print(f"  ✓ {output_path}: {len(df_concat):,} registros, {df_concat['veiculo_id'].nunique()} veículos")
        if verbose:
            print()
    if verbose:
        print("=" * 80)
        print("✅ PROCESSAMENTO CONCLUÍDO")
        print("=" * 80)
        print()

def _merge_sequential_windows(
    df_window_vehicles: pl.DataFrame,
    df_window_original: pl.DataFrame,
    verbose: bool = True
) -> pl.DataFrame:
    """
    Agrupa janelas sequenciais por veículo
    """
    if verbose:
        print("  🔗 Agrupando janelas sequenciais...")
    df_pd = df_window_vehicles.to_pandas()
    df_window_pd = df_window_original.to_pandas()
    df_pd = df_pd.sort_values(['veiculo_id', 'window'])
    grouped_data = []
    for vehicle_id, group in df_pd.groupby('veiculo_id'):
        windows = sorted(group['window'].tolist())
        # Identificar grupos sequenciais
        groups = []
        current_group = [windows[0]]
        for i in range(1, len(windows)):
            if windows[i] == windows[i-1] + 1:
                current_group.append(windows[i])
            else:
                groups.append(current_group)
                current_group = [windows[i]]
        groups.append(current_group)
        # Criar entradas
        for group_windows in groups:
            first_window = group_windows[0]
            last_window = group_windows[-1]
            first_window_data = df_window_pd[df_window_pd['window'] == first_window].iloc[0]
            dt_inicio = first_window_data['dt_inicio']
            last_window_data = df_window_pd[df_window_pd['window'] == last_window].iloc[0]
            dt_fim = last_window_data['dt_fim']
            if len(group_windows) == 1:
                window_label = str(group_windows[0])
            else:
                window_label = '-'.join(map(str, group_windows))
            grouped_data.append({
                'veiculo_id': vehicle_id,
                'window_group': window_label,
                'dt_inicio': dt_inicio,
                'dt_fim': dt_fim
            })
    df_grouped = pl.DataFrame(grouped_data)
    if verbose:
        n_original = len(df_pd)
        n_grouped = len(df_grouped)
        reduction = ((n_original - n_grouped) / n_original * 100) if n_original > 0 else 0
        print(f"    ✓ {n_original} → {n_grouped} grupos (redução: {reduction:.1f}%)")
        # Distribuição
        group_sizes = []
        for item in grouped_data:
            window = item['window_group']
            size = len(window.split('-')) if '-' in window else 1
            group_sizes.append(size)
        size_counts = Counter(group_sizes)
        print(f"    Distribuição:")
        for size in sorted(size_counts.keys()):
            count = size_counts[size]
            print(f"      {size} janela(s): {count} grupos")
    return df_grouped
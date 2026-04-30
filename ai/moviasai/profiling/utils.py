import polars as pl

from datetime import timedelta

from typing import List, Tuple, Dict

import time


def compute_effective_period(df: pl.DataFrame, target: str) -> pl.DataFrame:
    """
    Calcula período efetivo por veículo.

    Retorna DataFrame com colunas: veiculo_id, dt_inicio, dt_fim
    """
    return (
        df.filter(pl.col(target) > 0)
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


class WeeklyActivitySlicer:
    """
    Construtor de janelas temporais com critério de semanas ativas
    
    Cria janelas temporais onde TODAS as semanas devem ter atividade.
    Uma semana é considerada ativa se tem pelo menos um dia com target > 0.
    
    Usa processamento vetorizado para máxima performance.
    
    Attributes:
    -----------
    target : str
        Target: ``'km'`` ou ``'h'``
    col_target : str
        Nome da coluna alvo (``f'{target}_dia_clean'``)
    window_size : int
        Tamanho da janela em dias (deve ser múltiplo de 7)
    stride : int
        Passo entre janelas consecutivas (em dias)
    max_gap : int
        Máximo de dias de gap permitido entre registros
    n_weeks : int
        Número de semanas na janela (calculado automaticamente)
    """
    
    def __init__(
        self,
        target: str,
        window_size: int,
        stride: int,
        max_gap: int = 7
    ):
        """
        Inicializa o construtor de janelas
        
        Parameters:
        -----------
        target : str
            Target: ``'km'`` ou ``'h'``
        """
        if max_gap < 1:
            raise ValueError(f"max_gap deve ser >= 1. Recebido: {max_gap}")
        self.target = target
        self.col_target = f"{target}_dia_clean"
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
            df.filter(pl.col(self.col_target) > 0).select(['veiculo_id', 'data']),
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
        df_valid = df.filter(pl.col(self.col_target) > 0).sort(['veiculo_id', 'data'])
        
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
        df_active = df.filter(pl.col(self.col_target) > 0)
        
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
            'veiculo_id', 'window', 'data', self.col_target
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
        ...     target='km',
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
            print(f"   Max gap: {self.max_gap}")
        
        total_start = time.time()
        
        # 1. Identificar períodos contínuos
        if verbose:
            print("\n📍 Etapa 1: Identificando períodos contínuos...")
        
        df_periods = self._identify_continuous_periods(df)
        if len(df_periods) == 0:
            return pl.DataFrame(), pl.DataFrame()

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


class DatasetMerger:
    """
    Classe para merge de datasets formatados com raw data
    
    Responsabilidades:
    - Extrair chaves (veiculo_id, data) de datasets formatados
    - Fazer join com raw data
    - Validar e tratar valores nulos
    - Salvar resultado (opcional)
    
    Examples
    --------
    # Uso básico com DataFrames
    >>> merger = DatasetMerger(verbose=True)
    >>> df_merged = merger.merge_from_dataframes(
    ...     df_raw=df_raw,
    ...     formatted_dfs={'km_dia_clean': df_km, 'h_dia_clean': df_h}
    ... )
    
    # Uso com arquivos
    >>> df_merged = merger.merge_from_files(
    ...     path_raw_data='raw.csv',
    ...     path_km='km_formatted.csv',
    ...     path_h='h_formatted.csv',
    ...     output_dir='output'
    ... )
    """
    
    def __init__(self, verbose: bool = True):
        """
        Parameters
        ----------
        verbose : bool, default=True
            Se True, imprime informações durante o processo
        """
        self.verbose = verbose
    
    def _print(self, message: str):
        """Imprime mensagem se verbose=True"""
        if self.verbose:
            print(message)
    
    def _extract_keys(
        self,
        formatted_dfs: Dict[str, pl.DataFrame]
    ) -> pl.DataFrame:
        """
        Extrai e concatena chaves (veiculo_id, data) de datasets formatados
        
        Parameters
        ----------
        formatted_dfs : dict
            Dicionário {target: df_formatted}
        
        Returns
        -------
        pl.DataFrame
            DataFrame com chaves únicas: veiculo_id, data
        """
        self._print("Extraindo chaves dos datasets formatados...")
        
        dfs_keys = []
        
        for target, df in formatted_dfs.items():
            df_keys = df.select(['veiculo_id', 'data'])
            dfs_keys.append(df_keys)
            self._print(f"  • {target}: {len(df_keys):,} registros")
        
        # Concatenar
        self._print("\nConcatenando chaves...")
        df_keys = pl.concat(dfs_keys)
        self._print(f"  Total após concatenação: {len(df_keys):,}")
        
        # Remover duplicatas e ordenar
        df_keys = df_keys.unique(subset=['veiculo_id', 'data']).sort(['veiculo_id', 'data'])
        
        self._print(f"  Chaves únicas: {len(df_keys):,}")
        self._print(f"  Veículos únicos: {df_keys['veiculo_id'].n_unique()}")
        self._print("")
        
        return df_keys
    
    def _join_with_raw(
        self,
        df_keys: pl.DataFrame,
        df_raw: pl.DataFrame,
        targets: List[str]
    ) -> pl.DataFrame:
        """
        Faz join das chaves com raw data
        
        Parameters
        ----------
        df_keys : pl.DataFrame
            DataFrame com chaves: veiculo_id, data
        df_raw : pl.DataFrame
            DataFrame raw com todas as colunas
        targets : list
            Lista de targets a extrair do raw
        
        Returns
        -------
        pl.DataFrame
            DataFrame merged
        """
        self._print("Fazendo join com raw data...")
        
        cols_to_select = ['veiculo_id', 'data'] + targets
        
        df_merged = df_keys.join(
            df_raw.select(cols_to_select),
            on=['veiculo_id', 'data'],
            how='left'
        )
        
        return df_merged
    
    def _validate_and_fill_nulls(
        self,
        df_merged: pl.DataFrame,
        targets: List[str],
        fill_value: float = 0.0
    ) -> pl.DataFrame:
        """
        Valida e trata valores nulos
        
        Parameters
        ----------
        df_merged : pl.DataFrame
            DataFrame merged
        targets : list
            Lista de targets a verificar
        fill_value : float, default=0.0
            Valor para preencher nulos
        
        Returns
        -------
        pl.DataFrame
            DataFrame com nulos tratados
        """
        self._print("\nVerificando valores nulos:")
        
        has_nulls = False
        
        for target in targets:
            n_nulls = df_merged[target].null_count()
            
            if n_nulls > 0:
                has_nulls = True
                pct_null = n_nulls / len(df_merged) * 100
                self._print(f"  ⚠️  {target}: {n_nulls:,} nulos ({pct_null:.2f}%)")
            else:
                self._print(f"  ✓ {target}: sem nulos")
        
        if has_nulls:
            self._print(f"\n⚠️  Preenchendo nulos com {fill_value}")
            
            fill_exprs = [pl.col(t).fill_null(fill_value) for t in targets]
            df_merged = df_merged.with_columns(fill_exprs)
        
        return df_merged
    
    def _print_summary(self, df_merged: pl.DataFrame):
        """Imprime resumo do dataset merged"""
        self._print(f"\n✅ Dataset merged:")
        self._print(f"  • Registros: {len(df_merged):,}")
        self._print(f"  • Veículos: {df_merged['veiculo_id'].n_unique()}")
        self._print(f"  • Período: {df_merged['data'].min()} a {df_merged['data'].max()}")
        self._print(f"  • Colunas: {', '.join(df_merged.columns)}")
    
    def merge_from_dataframes(
        self,
        df_raw: pl.DataFrame,
        formatted_dfs: Dict[str, pl.DataFrame],
        fill_nulls: bool = True,
        fill_value: float = 0.0
    ) -> pl.DataFrame:
        """
        Merge usando DataFrames em memória
        
        Parameters
        ----------
        df_raw : pl.DataFrame
            DataFrame raw com todas as colunas
        formatted_dfs : dict
            Dicionário {target: df_formatted}
            Ex: {'km_dia_clean': df_km, 'h_dia_clean': df_h}
        fill_nulls : bool, default=True
            Se True, preenche nulos com fill_value
        fill_value : float, default=0.0
            Valor para preencher nulos
        
        Returns
        -------
        pl.DataFrame
            DataFrame merged
        """
        if self.verbose:
            print("="*80)
            print("MERGE DATA (FROM DATAFRAMES)")
            print("="*80)
            print()
            print(f"Targets: {', '.join(formatted_dfs.keys())}")
            print()
        
        targets = list(formatted_dfs.keys())
        
        # 1. Extrair chaves
        df_keys = self._extract_keys(formatted_dfs)
        
        # 2. Join com raw
        df_merged = self._join_with_raw(df_keys, df_raw, targets)
        
        # 3. Validar e tratar nulos
        if fill_nulls:
            df_merged = self._validate_and_fill_nulls(df_merged, targets, fill_value)
        
        # 4. Resumo
        self._print_summary(df_merged)
        
        if self.verbose:
            print()
            print("="*80)
            print()
        
        return df_merged


def merge_datasets(
    df_raw: pl.DataFrame,
    df_km: pl.DataFrame,
    df_h: pl.DataFrame,
    verbose: bool = True
) -> pl.DataFrame:
    """
    Conveniência para merge KM + H
    
    Wrapper para DatasetMerger.merge_from_dataframes()
    """
    merger = DatasetMerger(verbose=verbose)
    
    return merger.merge_from_dataframes(
        df_raw=df_raw,
        formatted_dfs={
            'km_dia_clean': df_km,
            'h_dia_clean': df_h
        }
    )

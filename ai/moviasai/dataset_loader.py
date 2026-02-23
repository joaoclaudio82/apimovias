import pandas as pd
import pickle
import hashlib
import re
import time

from pathlib import Path
from darts import TimeSeries
from functools import reduce
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import warnings
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from vehicle_profile import VehicleProfile


@dataclass
class Window:
    """Representa uma janela temporal"""
    series: List[TimeSeries]
    input_series: List[TimeSeries]
    output_series: List[TimeSeries]
    past_covariates: Optional[List[TimeSeries]]
    future_covariates: Optional[List[TimeSeries]]
    input_past_covariates: Optional[List[TimeSeries]]
    output_future_covariates: Optional[List[TimeSeries]]
    ids: List[int]
    
    def __post_init__(self):
        self._id_to_idx = {id_value: idx for idx, id_value in enumerate(self.ids)}
        self._validate()
    
    def _validate(self):
        n_series = len(self.series)
        if len(self.input_series) != n_series:
            raise ValueError(f"Inconsistente: series={n_series}, input_series={len(self.input_series)}")
        if len(self.output_series) != n_series:
            raise ValueError(f"Inconsistente: series={n_series}, output_series={len(self.output_series)}")
        if len(self.ids) != n_series:
            raise ValueError(f"Inconsistente: series={n_series}, ids={len(self.ids)}")
    
    def get_series_by_id(self, id_value: int, series_name: str = 'series') -> TimeSeries:
        idx = self._id_to_idx.get(id_value)
        if idx is None:
            raise ValueError(f'Veículo {id_value} não encontrado')
        series_list = getattr(self, series_name)
        if series_list is None:
            raise ValueError(f"Série '{series_name}' não disponível")
        return series_list[idx]
    
    def __len__(self) -> int:
        return len(self.series)
    
    def __repr__(self) -> str:
        return (
            f"Window(n_series={len(self.series)}, "
            f"ids={self.ids[:5]},{'...' if len(self.ids) > 5 else ''}, "
            f"has_past_cov={self.past_covariates is not None})"
        )
    

class SlicedDataset:
    """
    Dataset fatiado contendo múltiplas janelas temporais
    
    Esta classe agrega múltiplas janelas (Window) em um único dataset,
    permitindo acesso unificado às séries temporais de todas as janelas.
    
    As listas de séries são concatenadas sob demanda (lazy evaluation)
    quando as properties são acessadas, enquanto os mapeamentos de IDs
    são construídos na inicialização para acesso rápido.
    
    Attributes
    ----------
    windows : List[Window]
        Lista de janelas temporais
    _id_to_window_idx : Dict[int, List[tuple]]
        Mapeamento de ID para lista de (window_idx, series_idx)
    _n_total_series : int
        Número total de séries em todas as janelas
    
    Examples
    --------
    Criar dataset com múltiplas janelas:
    
    >>> window1 = Window(...)  # Janela 1 com IDs [1, 2, 3]
    >>> window2 = Window(...)  # Janela 2 com IDs [1, 4, 5]
    >>> window3 = Window(...)  # Janela 3 com IDs [2, 6, 7]
    >>> 
    >>> dataset = SlicedDataset([window1, window2, window3])
    >>> print(dataset)
    SlicedDataset(
      n_windows=3,
      n_total_series=9,
      n_unique_ids=7,
      ids=[1, 2, 3, 4, 5, ...]
    )
    
    Acessar séries agregadas:
    
    >>> all_series = dataset.series  # Concatena séries de todas as janelas
    >>> print(len(all_series))
    9
    
    Buscar séries por ID (pode estar em múltiplas janelas):
    
    >>> series_list = dataset.get_series_by_id(1)  # ID 1 está em window1 e window2
    >>> print(len(series_list))
    2
    """
    
    def __init__(self, windows: List[Window]):
        """
        Inicializa o dataset fatiado
        
        Parameters
        ----------
        windows : List[Window]
            Lista de janelas temporais a serem agregadas
            
        Raises
        ------
        ValueError
            Se a lista de janelas estiver vazia
            
        Examples
        --------
        >>> dataset = SlicedDataset([window1, window2, window3])
        """
        if not windows:
            raise ValueError("Lista de janelas não pode estar vazia")
        
        self.windows = windows
        
        # Construir mapeamentos na inicialização
        self._id_to_window_idx = self._build_id_mapping()
        self._n_total_series = sum(len(w) for w in windows)
        
        # Verificar consistência de covariáveis
        self._has_past_covariates = all(
            w.past_covariates is not None for w in windows
        ) if any(w.past_covariates is not None for w in windows) else False
        
        self._has_future_covariates = all(
            w.future_covariates is not None for w in windows
        ) if any(w.future_covariates is not None for w in windows) else False
        
        self._has_input_past_covariates = all(
            w.input_past_covariates is not None for w in windows
        ) if any(w.input_past_covariates is not None for w in windows) else False
        
        self._has_output_future_covariates = all(
            w.output_future_covariates is not None for w in windows
        ) if any(w.output_future_covariates is not None for w in windows) else False
    
    def _build_id_mapping(self) -> Dict[int, List[tuple]]:
        """
        Constrói mapeamento de ID para lista de (window_idx, series_idx)
        
        Permite que um ID apareça em múltiplas janelas, mapeando para
        todas as ocorrências.
        
        Returns
        -------
        Dict[int, List[tuple]]
            Dicionário onde cada ID mapeia para lista de tuplas
            (índice_da_janela, índice_na_janela)
            
        Examples
        --------
        >>> # ID 1 aparece na janela 0 (índice 0) e janela 2 (índice 1)
        >>> {1: [(0, 0), (2, 1)], 2: [(0, 1)], ...}
        """
        id_mapping = {}
        
        for window_idx, window in enumerate(self.windows):
            for series_idx, id_value in enumerate(window.ids):
                if id_value not in id_mapping:
                    id_mapping[id_value] = []
                id_mapping[id_value].append((window_idx, series_idx))
        
        return id_mapping
    
    # ========================================================================
    # PROPERTIES - Concatenação sob demanda
    # ========================================================================
    @property
    def ids(self) -> List[int]:
        """
        Retorna o id das séries de todas as janelas concatenadas
        Obs. pode haver valores repetidos. 
        
        Returns
        -------
        List[TimeSeries]
            Lista todos os ids
            
        Examples
        --------
        >>> all_ids = dataset.ids
        >>> print(len(all_ids))
        150  # Total de séries em todas as janelas
        """


        result = []
        for window in self.windows:
            result.extend(window.ids)
        return result
    
    @property
    def series(self) -> List[TimeSeries]:
        """
        Retorna todas as séries de todas as janelas concatenadas
        
        Returns
        -------
        List[TimeSeries]
            Lista concatenada de todas as séries
            
        Examples
        --------
        >>> all_series = dataset.series
        >>> print(len(all_series))
        150  # Total de séries em todas as janelas
        """
        result = []
        for window in self.windows:
            result.extend(window.series)
        return result
    
    @property
    def input_series(self) -> List[TimeSeries]:
        """
        Retorna todas as séries de entrada de todas as janelas concatenadas
        
        Returns
        -------
        List[TimeSeries]
            Lista concatenada de todas as séries de entrada
        """
        result = []
        for window in self.windows:
            result.extend(window.input_series)
        return result
    
    @property
    def output_series(self) -> List[TimeSeries]:
        """
        Retorna todas as séries de saída de todas as janelas concatenadas
        
        Returns
        -------
        List[TimeSeries]
            Lista concatenada de todas as séries de saída
        """
        result = []
        for window in self.windows:
            result.extend(window.output_series)
        return result
    
    @property
    def past_covariates(self) -> Optional[List[TimeSeries]]:
        """
        Retorna todas as covariáveis passadas de todas as janelas concatenadas
        
        Returns
        -------
        Optional[List[TimeSeries]]
            Lista concatenada ou None se não houver covariáveis passadas
        """
        if not self._has_past_covariates:
            return None
        
        result = []
        for window in self.windows:
            if window.past_covariates is not None:
                result.extend(window.past_covariates)
        return result
    
    @property
    def future_covariates(self) -> Optional[List[TimeSeries]]:
        """
        Retorna todas as covariáveis futuras de todas as janelas concatenadas
        
        Returns
        -------
        Optional[List[TimeSeries]]
            Lista concatenada ou None se não houver covariáveis futuras
        """
        if not self._has_future_covariates:
            return None
        
        result = []
        for window in self.windows:
            if window.future_covariates is not None:
                result.extend(window.future_covariates)
        return result
    
    @property
    def input_past_covariates(self) -> Optional[List[TimeSeries]]:
        """
        Retorna todas as covariáveis passadas de entrada concatenadas
        
        Returns
        -------
        Optional[List[TimeSeries]]
            Lista concatenada ou None se não disponível
        """
        if not self._has_input_past_covariates:
            return None
        
        result = []
        for window in self.windows:
            if window.input_past_covariates is not None:
                result.extend(window.input_past_covariates)
        return result
    
    @property
    def output_future_covariates(self) -> Optional[List[TimeSeries]]:
        """
        Retorna todas as covariáveis futuras de saída concatenadas
        
        Returns
        -------
        Optional[List[TimeSeries]]
            Lista concatenada ou None se não disponível
        """
        if not self._has_output_future_covariates:
            return None
        
        result = []
        for window in self.windows:
            if window.output_future_covariates is not None:
                result.extend(window.output_future_covariates)
        return result
    
    # ========================================================================
    # MÉTODOS DE ACESSO POR ID
    # ========================================================================
    def get_series_by_id(
        self, 
        id_value: int, 
        series_name: str = 'series'
    ) -> List[TimeSeries]:
        """
        Retorna todas as séries de um ID em todas as janelas
        
        Parameters
        ----------
        id_value : int
            ID do veículo
        series_name : str, default='series'
            Nome da série a retornar
            Opções: 'series', 'input_series', 'output_series', 'past_covariates',
                   'future_covariates', 'input_past_covariates', 'output_future_covariates'
        
        Returns
        -------
        List[TimeSeries]
            Lista de séries temporais do ID (pode ter múltiplas se ID aparece
            em múltiplas janelas)
        
        Raises
        ------
        ValueError
            Se o ID não for encontrado ou se o nome da série for inválido
            
        Examples
        --------
        >>> # ID 1 aparece em 2 janelas
        >>> series_list = dataset.get_series_by_id(1)
        >>> print(len(series_list))
        2
        
        >>> # Buscar séries de entrada
        >>> input_list = dataset.get_series_by_id(1, series_name='input_series')
        """
        if id_value not in self._id_to_window_idx:
            raise ValueError(
                f'Veículo com ID {id_value} não encontrado. '
                f'IDs disponíveis: {sorted(self.get_unique_ids())}'
            )
        
        valid_series = [
            'series', 'input_series', 'output_series',
            'past_covariates', 'future_covariates',
            'input_past_covariates', 'output_future_covariates'
        ]
        
        if series_name not in valid_series:
            raise ValueError(
                f"Nome de série inválido: '{series_name}'. "
                f"Opções válidas: {', '.join(valid_series)}"
            )
        
        result = []
        for window_idx, series_idx in self._id_to_window_idx[id_value]:
            window = self.windows[window_idx]
            series_list = getattr(window, series_name)
            
            if series_list is None:
                raise ValueError(
                    f"Série '{series_name}' não está disponível na janela {window_idx}"
                )
            
            result.append(series_list[series_idx])
        
        return result
    
    def get_series_indices(self, id_value: int) -> List[tuple]:
        """
        Retorna todos os índices (window_idx, series_idx) para um ID
        
        Parameters
        ----------
        id_value : int
            ID do veículo
        
        Returns
        -------
        List[tuple]
            Lista de tuplas (window_idx, series_idx)
        
        Raises
        ------
        KeyError
            Se o ID não for encontrado
            
        Examples
        --------
        >>> indices = dataset.get_series_indices(1)
        >>> print(indices)
        [(0, 0), (2, 5)]  # ID 1 na janela 0 (índice 0) e janela 2 (índice 5)
        """
        if id_value not in self._id_to_window_idx:
            raise KeyError(
                f'Veículo com ID {id_value} não encontrado. '
                f'IDs disponíveis: {sorted(self.get_unique_ids())}'
            )
        return self._id_to_window_idx[id_value]
    
    def get_series_idx(self, id_value: int, occurrence: int = 0) -> tuple:
        """
        Retorna índice (window_idx, series_idx) de uma ocorrência específica
        
        Parameters
        ----------
        id_value : int
            ID do veículo
        occurrence : int, default=0
            Qual ocorrência do ID retornar (0 = primeira)
        
        Returns
        -------
        tuple
            Tupla (window_idx, series_idx)
        
        Raises
        ------
        KeyError
            Se o ID não for encontrado
        IndexError
            Se a ocorrência solicitada não existir
            
        Examples
        --------
        >>> # ID 1 aparece em 2 janelas
        >>> idx = dataset.get_series_idx(1, occurrence=0)
        >>> print(idx)
        (0, 0)  # Janela 0, índice 0
        >>> 
        >>> idx = dataset.get_series_idx(1, occurrence=1)
        >>> print(idx)
        (2, 5)  # Janela 2, índice 5
        """
        if id_value not in self._id_to_window_idx:
            raise KeyError(
                f'Veículo com ID {id_value} não encontrado. '
                f'IDs disponíveis: {sorted(self.get_unique_ids())}'
            )
        
        indices = self._id_to_window_idx[id_value]
        
        if occurrence >= len(indices):
            raise IndexError(
                f'Ocorrência {occurrence} não existe para ID {id_value}. '
                f'Ocorrências disponíveis: 0 a {len(indices)-1}'
            )
        
        return indices[occurrence]
    
    def has_id(self, id_value: int) -> bool:
        """
        Verifica se um ID existe no dataset
        
        Parameters
        ----------
        id_value : int
            ID do veículo
        
        Returns
        -------
        bool
            True se o ID existe, False caso contrário
            
        Examples
        --------
        >>> dataset.has_id(1)
        True
        >>> dataset.has_id(999)
        False
        """
        return id_value in self._id_to_window_idx
    
    def get_id_count(self, id_value: int) -> int:
        """
        Retorna quantas vezes um ID aparece no dataset
        
        Parameters
        ----------
        id_value : int
            ID do veículo
        
        Returns
        -------
        int
            Número de ocorrências do ID (0 se não encontrado)
            
        Examples
        --------
        >>> dataset.get_id_count(1)
        2  # ID 1 aparece em 2 janelas
        >>> dataset.get_id_count(999)
        0  # ID não existe
        """
        if id_value not in self._id_to_window_idx:
            return 0
        return len(self._id_to_window_idx[id_value])
    
    def get_unique_ids(self) -> List[int]:
        """
        Retorna lista de IDs únicos no dataset
        
        Returns
        -------
        List[int]
            Lista ordenada de IDs únicos
            
        Examples
        --------
        >>> dataset.get_unique_ids()
        [1, 2, 3, 4, 5, 6, 7]
        """
        return sorted(self._id_to_window_idx.keys())
    
    def get_duplicate_ids(self) -> Dict[int, int]:
        """
        Retorna IDs que aparecem em múltiplas janelas
        
        Returns
        -------
        Dict[int, int]
            Dicionário {id: contagem} para IDs com contagem > 1
            
        Examples
        --------
        >>> dataset.get_duplicate_ids()
        {1: 2, 2: 3}  # ID 1 aparece 2 vezes, ID 2 aparece 3 vezes
        """
        return {
            id_value: count
            for id_value, indices in self._id_to_window_idx.items()
            if (count := len(indices)) > 1
        }
    
    def has_duplicates(self) -> bool:
        """
        Verifica se existem IDs em múltiplas janelas
        
        Returns
        -------
        bool
            True se houver IDs duplicados, False caso contrário
            
        Examples
        --------
        >>> dataset.has_duplicates()
        True
        """
        return any(len(indices) > 1 for indices in self._id_to_window_idx.values())
    
    def __len__(self) -> int:
        """Retorna o número total de séries no dataset"""
        return self._n_total_series
    
    def __repr__(self) -> str:
        """Representação em string do objeto"""
        n_unique = len(self._id_to_window_idx)
        duplicates_info = ""
        
        if self.has_duplicates():
            n_duplicates = self._n_total_series - n_unique
            duplicates_info = f", duplicates={n_duplicates}"
        
        all_ids = self.get_unique_ids()
        ids_preview = all_ids[:5]
        ids_str = str(ids_preview) + ('...' if len(all_ids) > 5 else '')
        
        return (
            f"SlicedDataset(\n"
            f"  n_windows={len(self.windows)},\n"
            f"  n_total_series={self._n_total_series},\n"
            f"  n_unique_ids={n_unique},{duplicates_info},\n"
            f"  ids={ids_str},\n"
            f"  has_past_covariates={self._has_past_covariates},\n"
            f"  has_future_covariates={self._has_future_covariates}\n"
            f")"
        )


class DatasetLoader:
    """
    Carregador otimizado com batch processing de perfis
    """
    
    def __init__(
        self,
        dataset_dir: str,
        vp: VehicleProfile,
        past_covariates_extractor: Optional[List] = None,
        future_covariates_extractor: Optional[List] = None,
        verbose: bool = True
    ):
        """
        Inicializa o carregador
        
        Parameters:
        -----------
        dataset_dir : str
            Diretório do dataset
        vp : VehicleProfile
            Perfil de veículos
        past_covariates_extractor : List, optional
            Extratores de covariáveis passadas adicionais
        future_covariates_extractor : List, optional
            Extratores de covariáveis futuras
        verbose : bool, optional
            Se True, imprime logs
        """
        self.dataset_dir = Path(dataset_dir)
        self.vp = vp
        self.verbose = verbose
        self.past_covariates_extractor = past_covariates_extractor or []
        self.future_covariates_extractor = future_covariates_extractor or []
        
        self._extract_params_from_dirname()
        
        if self.target != vp.target:
            raise ValueError(f"Target incompatível: dataset={self.target}, VehicleProfile={vp.target}")

    @staticmethod 
    def _extract_target(dirname):
        match = re.search(r'dataset_([^_]+(?:_[^_]+)*)(?=_mindays)', dirname)
        if match:
            return match.group(1)
        
        match = re.search(r'dataset_([^_]+(?:_[^_]+)*)(?=_hist)', dirname)
        if match:
            return match.group(1)
        
        return None
    
    def _extract_params_from_dirname(self):
        """Extrai parâmetros do diretório"""
        dirname = self.dataset_dir.name
        
        if self.verbose:
            print("=" * 80)
            print("🔍 EXTRAINDO PARÂMETROS DO DIRETÓRIO")
            print("=" * 80)
            print(f"Diretório: {dirname}\n")
        
        # Target
        target = self._extract_target(dirname)
        if target is not None:
            self.target = target
            if self.verbose:
                print(f"✓ Target: '{self.target}'")
        else:
            raise ValueError(f"Não foi possível extrair target: {dirname}")
        
        # history_size
        match_hist = re.search(r'hist(\d+)', dirname)
        if match_hist:
            self.history_size = int(match_hist.group(1))
            if self.verbose:
                print(f"✓ History_size: {self.history_size}")
        else:
            raise ValueError("Não foi possível extrair history_size")
        
        # forecast_horizon
        match_fh = re.search(r'fh(\d+)', dirname)
        if match_fh:
            self.forecast_horizon = int(match_fh.group(1))
            if self.verbose:
                print(f"✓ Forecast_horizon: {self.forecast_horizon}")
        else:
            raise ValueError("Não foi possível extrair forecast_horizon")
        
        if self.verbose:
            print()
    
    def _load_split_data(self, split: str) -> pd.DataFrame:
        """Carrega dados de um split"""
        data_file = self.dataset_dir / f'{split}.csv'
        
        if not data_file.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {data_file}")
        
        if self.verbose:
            print("=" * 80)
            print(f"📂 CARREGANDO DADOS ({split.upper()})")
            print("=" * 80)
            print(f"Arquivo: {data_file.name}\n")
        
        start = time.time()
        
        df_data = pd.read_csv(data_file, parse_dates=['data'])
        df_data = df_data.sort_values(['veiculo_id', 'window', 'data']).reset_index(drop=True)
        
        elapsed = time.time() - start
        
        if self.verbose:
            print(f"✓ Carregado em {elapsed:.2f}s")
            print(f"  - Registros: {len(df_data):,}")
            print(f"  - Veículos: {df_data['veiculo_id'].nunique()}")
            print(f"  - Janelas: {df_data['window'].nunique()}")
            print(f"  - Período: {df_data['data'].min()} a {df_data['data'].max()}")
            print(f"  - Colunas: {list(df_data.columns)}")
            print("=" * 80 + "\n")
        
        return df_data
    
    def _compute_cache_hash(self, df_data: pd.DataFrame, split: str) -> str:
        """Computa hash para cache"""
        config_str = (
            f"{self.target}_{self.history_size}_{self.forecast_horizon}_"
            f"{len(self.past_covariates_extractor)}_{len(self.future_covariates_extractor)}_"
            f"{split}"
        )
        
        df_hash = hashlib.md5(
            pd.util.hash_pandas_object(df_data, index=True).values
        ).hexdigest()
        
        vp_hash = hashlib.md5(str(len(self.vp.df_versions)).encode()).hexdigest()[:8]
        
        combined = f"{df_hash}_{config_str}_{vp_hash}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    @staticmethod
    def _process_single_group(
        group_info: dict,
        profiles_cache: dict,
        vp: 'VehicleProfile',
        target_col: str
    ) -> Tuple[Optional[TimeSeries], Optional[TimeSeries], Optional[Tuple]]:
        """
        Processa um grupo (paralelizável)
        
        Returns:
        --------
        tuple
            (ts, ts_daily, group_key) ou (None, None, None) se erro
        """
        vehicle_id = group_info['vehicle_id']
        values_raw = group_info['values_raw']
        time_index = group_info['time_index']
        history_end_date = group_info['history_end_date']
        group_key = group_info['group_key']
        
        try:
            cache_key = (vehicle_id, history_end_date)
            
            if cache_key not in profiles_cache:
                return None, None, None
            
            profile_row = profiles_cache[cache_key]
            upper = profile_row['upper']
            
            # 1. NORMALIZAR
            clipped = np.clip(values_raw, 0, upper)
            values_scaled = (clipped / upper if upper > 0 else np.zeros_like(clipped)).reshape(-1, 1)
            
            # 2. STATIC COVARIATES
            static_cov = pd.DataFrame([{
                feat: profile_row[feat]
                for feat in vp.segmentation_features
            }])
            
            # 3. TIMESERIES
            ts = TimeSeries.from_times_and_values(
                times=time_index,
                values=values_scaled,
                static_covariates=static_cov,
                freq='D',
                fill_missing_dates=False,
                columns=[f'{target_col}_scaled']
            )
            
            # 4. DAILY FEATURES
            n_dates = len(time_index)
            n_features = len(vp.weekday_base_features)
            daily_features_matrix = np.zeros((n_dates, n_features))
            
            weekdays = [date.weekday() + 1 for date in time_index]
            
            for i, weekday in enumerate(weekdays):
                for j, feat in enumerate(vp.weekday_base_features):
                    col_name = f'day_{weekday}_{feat}'
                    daily_features_matrix[i, j] = profile_row[col_name]
            
            # 5. TIMESERIES DAILY
            ts_daily = TimeSeries.from_times_and_values(
                times=time_index,
                values=daily_features_matrix,
                freq='D',
                fill_missing_dates=False,
                columns=vp.weekday_base_features
            )
            
            return ts, ts_daily, group_key
        
        except Exception as e:
            warnings.warn(f"Erro ao processar veículo {vehicle_id}: {e}")
            return None, None, None
    
    def _create_timeseries_with_profile(
        self,
        df_data: pd.DataFrame,
        group_cols: List[str],
        time_col: str,
        target_col: str,
        vp: VehicleProfile,
        history_size: int
    ) -> Tuple[List[TimeSeries], List[Tuple], List[TimeSeries]]:
        start = time.time()
        
        df_sorted = df_data.sort_values(group_cols + [time_col])
        grouped = df_sorted.groupby(group_cols, sort=False)
        
        # PASSO 1: Coletar informações
        if self.verbose:
            print("   🔄 Coletando informações dos grupos...")
        
        groups_info = []
        all_vehicle_ids = set()
        all_dates = set()
        
        for group_key, group_df in grouped:
            vehicle_id_idx = group_cols.index('veiculo_id')
            vehicle_id = int(group_key[vehicle_id_idx])
            
            values_raw = group_df[target_col].values
            time_index = pd.DatetimeIndex(group_df[time_col])
            history_end_date = time_index[history_size - 1]
            
            groups_info.append({
                'group_key': group_key,
                'vehicle_id': vehicle_id,
                'values_raw': values_raw,
                'time_index': time_index,
                'history_end_date': history_end_date
            })
            
            all_vehicle_ids.add(vehicle_id)
            all_dates.add(history_end_date)
        
        if self.verbose:
            elapsed = time.time() - start
            print(f"      ✓ {len(groups_info)} grupos coletados ({elapsed:.2f}s)")
        
        # PASSO 2: Buscar perfis em BATCH
        if self.verbose:
            print("   🔄 Buscando perfis em batch...")
        
        start_batch = time.time()
        
        unique_vehicle_ids = list(all_vehicle_ids)
        unique_dates = list(all_dates)
        
        if self.verbose:
            print(f"      Veículos únicos: {len(unique_vehicle_ids)}")
            print(f"      Datas únicas: {len(unique_dates)}")
        
        profiles_cache = vp.get_profiles_batch(
            vehicle_ids=unique_vehicle_ids,
            dates=unique_dates,
            normalized=True
        )
        
        elapsed_batch = time.time() - start_batch
        
        if self.verbose:
            print(f"      ✓ {len(profiles_cache)} perfis carregados ({elapsed_batch:.2f}s)")
        
        if self.verbose:
            print("   🔄 Criando TimeSeries...")
        
        start_create = time.time()
        
        series_list = []
        daily_covariates_list = []
        keys_list = []
        
        weekday_feat_cols = vp.weekday_base_features
        n_weekday_features = len(weekday_feat_cols)
        
        for group_info in tqdm(groups_info, desc="Criando séries", disable=not self.verbose):
            vehicle_id = group_info['vehicle_id']
            values_raw = group_info['values_raw']
            time_index = group_info['time_index']
            history_end_date = group_info['history_end_date']
            group_key = group_info['group_key']
            
            # Buscar no cache usando tupla (vehicle_id, date)
            cache_key = (vehicle_id, history_end_date)
            
            if cache_key not in profiles_cache:
                warnings.warn(f"Perfil não encontrado para veículo {vehicle_id}")
                continue
            
            try:
                profile_row = profiles_cache[cache_key]
                upper = profile_row['upper']
                
                # 1. NORMALIZAR
                clipped = np.clip(values_raw, 0, upper)
                values_scaled = (clipped / upper if upper > 0 else np.zeros_like(clipped)).reshape(-1, 1)
                
                # 2. STATIC COVARIATES
                static_cov = pd.DataFrame([{
                    feat: profile_row[feat]
                    for feat in vp.segmentation_features
                }])
                
                # 3. TIMESERIES
                ts = TimeSeries.from_times_and_values(
                    times=time_index,
                    values=values_scaled,
                    static_covariates=static_cov,
                    freq='D',
                    fill_missing_dates=False,
                    columns=[f'{target_col}_scaled']
                )
                
                # 4. DAILY FEATURES
                n_dates = len(time_index)
                daily_features_matrix = np.empty((n_dates, n_weekday_features))
                
                for i, date in enumerate(time_index):
                    weekday = date.weekday() + 1
                    for j, feat in enumerate(weekday_feat_cols):
                        daily_features_matrix[i, j] = profile_row[f'day_{weekday}_{feat}']
                
                # 5. TIMESERIES DAILY
                ts_daily = TimeSeries.from_times_and_values(
                    times=time_index,
                    values=daily_features_matrix,
                    freq='D',
                    fill_missing_dates=False,
                    columns=weekday_feat_cols
                )
                
                series_list.append(ts)
                daily_covariates_list.append(ts_daily)
                keys_list.append(group_key)
            
            except Exception as e:
                warnings.warn(f"Erro ao processar veículo {vehicle_id}: {e}")
                continue
        
        elapsed_create = time.time() - start_create
        total_elapsed = time.time() - start
        
        if self.verbose:
            print(f"      ✓ {len(series_list)} séries criadas ({elapsed_create:.2f}s)")
            print(f"      Tempo total: {total_elapsed:.2f}s")
        
        return series_list, keys_list, daily_covariates_list
    
    def _extract_covariates(
        self,
        series: List[TimeSeries],
        daily_covariates: List[TimeSeries],
        past: bool
    ) -> Tuple[Optional[List[TimeSeries]], Optional[List[TimeSeries]]]:
        """Extrai covariáveis combinando extratores + perfil diário"""
        extractors = self.past_covariates_extractor if past else self.future_covariates_extractor
        
        if past:
            all_covariates = daily_covariates.copy()
            
            if extractors:
                for i, ts in enumerate(series):
                    ts_covs = [extractor.extract(ts) for extractor in extractors]
                    
                    if len(ts_covs) == 1:
                        extractor_cov = ts_covs[0]
                    else:
                        extractor_cov = reduce(lambda x, y: x.stack(y), ts_covs)
                    
                    all_covariates[i] = all_covariates[i].stack(extractor_cov)
        
        else:
            if not extractors:
                return None, None
            
            all_covariates = []
            for ts in series:
                ts_covs = [extractor.extract(ts) for extractor in extractors]
                
                if len(ts_covs) == 1:
                    covariate = ts_covs[0]
                else:
                    covariate = reduce(lambda x, y: x.stack(y), ts_covs)
                
                all_covariates.append(covariate)
        
        covariates_test = []
        for covariate in all_covariates:
            if past:
                test_cov = covariate[:-self.forecast_horizon]
            else:
                test_cov = covariate[-self.forecast_horizon:]
            
            covariates_test.append(test_cov)
        
        return all_covariates, covariates_test
    
    def _get_train_test(
        self,
        series: List[TimeSeries]
    ) -> Tuple[List[TimeSeries], List[TimeSeries]]:
        """Divide em treino e teste"""
        train_series = []
        test_series = []
        
        for ts in series:
            if len(ts) < self.history_size + self.forecast_horizon:
                raise ValueError(f"Série muito curta: {len(ts)} < {self.history_size + self.forecast_horizon}")
            
            train_ts = ts[:-self.forecast_horizon]
            test_ts = ts[-self.forecast_horizon:]
            
            train_series.append(train_ts)
            test_series.append(test_ts)
        
        return train_series, test_series
    
    def _try_load_from_cache(
        self,
        df_data: pd.DataFrame,
        split: str,
        cache_dir: str
    ) -> Optional[List[Window]]:
        """Tenta carregar do cache"""
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        cache_hash = self._compute_cache_hash(df_data, split)
        cache_file = cache_path / f'windows_{split}_{cache_hash}.pkl'
        
        if not cache_file.exists():
            return None
        
        if self.verbose:
            print("=" * 80)
            print("📦 CARREGANDO JANELAS DO CACHE")
            print("=" * 80)
            print(f"Split: {split}")
            print(f"Cache: {cache_file.name}\n")
        
        try:
            with open(cache_file, 'rb') as f:
                windows = pickle.load(f)
            
            if self.verbose:
                print(f"✓ {len(windows)} janelas carregadas")
                print(f"Total de séries: {sum(len(w) for w in windows)}")
                print("=" * 80 + "\n")
            
            return windows
        
        except Exception as e:
            warnings.warn(f"Erro ao carregar cache: {e}. Recriando...")
            return None
    
    def _save_to_cache(
        self,
        windows: List[Window],
        df_data: pd.DataFrame,
        split: str,
        cache_dir: str
    ):
        """Salva no cache"""
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        cache_hash = self._compute_cache_hash(df_data, split)
        cache_file = cache_path / f'windows_{split}_{cache_hash}.pkl'
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(windows, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            if self.verbose:
                file_size = cache_file.stat().st_size / (1024 * 1024)
                print(f"\n💾 Cache salvo: {cache_file.name} ({file_size:.2f} MB)")
        
        except Exception as e:
            warnings.warn(f"Erro ao salvar cache: {e}")
    
    def load(
        self,
        split: str = 'train',
        return_windows: bool = False,
        use_cache: bool = True,
        cache_dir: str = '.cache/windows',
        force_reload: bool = False,
        split_dir: Optional[str] = None,
    ) -> Union['SlicedDataset', List[Window]]:
        """
        Carrega e processa dados de um split
        
        Parameters:
        -----------
        split : str, optional
            Split: 'train', 'val', 'test' (padrão: 'train')
        return_windows : bool, optional
            Se True, retorna List[Window] (padrão: False)
        use_cache : bool, optional
            Se True, usa cache (padrão: True)
        cache_dir : str, optional
            Diretório de cache (padrão: '.cache/windows')
        force_reload : bool, optional
            Se True, ignora cache (padrão: False)      
        Returns:
        --------
        Union[SlicedDataset, List[Window]]
            Janelas de dados
        """
        df_data = self._load_split_data(split)
        
        if use_cache and not force_reload:
            windows = self._try_load_from_cache(df_data, split, cache_dir)
            if windows is not None:
                return windows if return_windows else SlicedDataset(windows)
        
        required_cols = ['window', 'veiculo_id', 'data', self.target]
        missing = [col for col in required_cols if col not in df_data.columns]
        
        if missing:
            raise ValueError(f"Colunas faltando: {', '.join(missing)}")
        
        if self.verbose:
            print("=" * 80)
            print(f"🔄 CRIANDO JANELAS ({split.upper()})")
            print("=" * 80)
            print(f"Target: {self.target}")
            print(f"History: {self.history_size} dias")
            print(f"Forecast: {self.forecast_horizon} dias")
            print(f"Extratores: {len(self.past_covariates_extractor)} past, {len(self.future_covariates_extractor)} future")
            print(f"Perfil: {len(self.vp.segmentation_features)} seg + {len(self.vp.weekday_base_features)} daily")
            print()
        
        start_time = time.time()
        
        # Criar séries com batch + paralelização
        dataset_ts, group_keys, daily_covariates = self._create_timeseries_with_profile(
            df_data=df_data,
            group_cols=['window', 'veiculo_id'],
            time_col='data',
            target_col=self.target,
            vp=self.vp,
            history_size=self.history_size
        )
        
        if self.verbose:
            elapsed = time.time() - start_time
            print(f"✓ {len(dataset_ts)} séries criadas em {elapsed:.2f}s\n")
        
        df_group = pd.DataFrame(group_keys, columns=['window', 'veiculo_id'])
        
        windows = []
        unique_windows = sorted(df_group['window'].unique())
        
        for window_id in unique_windows:
            group_attr = df_group[df_group['window'] == window_id]
            start_idx = group_attr.index[0]
            end_idx = group_attr.index[-1] + 1
            
            group_series = dataset_ts[start_idx:end_idx]
            group_daily_cov = daily_covariates[start_idx:end_idx]
            group_ids = group_attr['veiculo_id'].to_list()
            
            train_series, test_series = self._get_train_test(group_series)
            
            past_covariates, test_past_covariates = self._extract_covariates(
                group_series, group_daily_cov, past=True
            )
            future_covariates, test_future_covariates = self._extract_covariates(
                group_series, group_daily_cov, past=False
            )
            
            window = Window(
                series=group_series,
                input_series=train_series,
                output_series=test_series,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
                input_past_covariates=test_past_covariates,
                output_future_covariates=test_future_covariates,
                ids=group_ids
            )
            
            windows.append(window)
            
            if self.verbose:
                print(f"   ✓ Janela {window_id}: {len(group_ids)} veículos")
        
        if use_cache:
            self._save_to_cache(windows, df_data, split, cache_dir)
        
        total_time = time.time() - start_time
        
        if self.verbose:
            print()
            print("=" * 80)
            print(f"✓ {len(windows)} janelas criadas em {total_time:.2f}s")
            print(f"Total de séries: {sum(len(w) for w in windows)}")
            print("=" * 80 + "\n")
        
        return windows if return_windows else SlicedDataset(windows)
    
    def __repr__(self) -> str:
        return (
            f"DatasetLoader(\n"
            f"  dataset_dir='{self.dataset_dir.name}',\n"
            f"  target='{self.target}',\n"
            f"  history_size={self.history_size},\n"
            f"  forecast_horizon={self.forecast_horizon},\n"
            f"  n_past_extractors={len(self.past_covariates_extractor)},\n"
            f"  n_future_extractors={len(self.future_covariates_extractor)},\n"
            f"  vehicle_profile=True\n"
            f")"
        )
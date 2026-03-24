# api/services/predictor_service.py

import sys
from typing import List, Dict, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from darts import TimeSeries

from darts.models import (
    DLinearModel,
    NLinearModel,
    TSMixerModel,
    TCNModel,
    NBEATSModel,
    NHiTSModel,
    LightGBMModel,
    XGBModel,
    LinearRegressionModel
)

from api.services.vehicle_profile_service import VehicleProfileService
from api.config.prediction_config import PredictorConfig
from moviasai.prediction import Predictor
from moviasai.covariates import is_weekday, week_of_month

logger = logging.getLogger(__name__)

# Mapeamento de nome para classe do modelo
MODEL_CLASS_MAP = {
    'DLinearModel': DLinearModel,
    'NLinearModel': NLinearModel,
    'TSMixerModel': TSMixerModel,
    'TCNModel': TCNModel,
    'NBEATSModel': NBEATSModel,
    'NHiTSModel': NHiTSModel,
    'LightGBMModel': LightGBMModel,
    'XGBModel': XGBModel,
    'LinearRegressionModel': LinearRegressionModel,
}


class PredictorService:
    """
    Service para gerenciar predições de veículos
    
    Responsabilidades:
    - Carregar modelos treinados do disco
    - Criar predictors com modelos únicos ou ensembles
    - Buscar perfis de veículos
    - Criar TimeSeries para predição
    - Executar predições em lote com paralelização
    
    Examples
    --------
    >>> async with AsyncSessionLocal() as session:
    ...     config = PredictorConfig.from_yaml('config/predictor_config.yaml')
    ...     service = PredictorService(session, config)
    ...     
    ...     # Predizer quando vai atingir 10.000 km
    ...     results = await service.predict_date_to_reach(
    ...         vehicle_ids=[1316, 18230],
    ...         target_values=[10000.0, 15000.0],
    ...         n_jobs=-1
    ...     )
    """
    
    def __init__(
        self,
        session: AsyncSession,
        config: PredictorConfig,
        profile_service: VehicleProfileService
    ):
        self.session = session
        self.config = config
        self.profile_service = profile_service
        
        # Cache de modelos carregados
        # Key: (category, segment, model_name)
        self._model_cache: Dict[Tuple[str, int, str], object] = {}
        
        # Cache de predictors
        # Key: (category, segment)
        self._predictor_cache: Dict[Tuple[str, int], Predictor] = {}
        self._register_custom_encoders()

    @staticmethod
    def _register_custom_encoders():
        """
        Registra encoders customizados no namespace global
        
        Necessário para que pickle consiga desserializar os modelos
        que foram salvos com essas funções.
        """
        import __main__
        
        # Registrar no __main__ para pickle encontrar
        __main__.week_of_month = week_of_month
        __main__.is_weekday = is_weekday
        
        # Registrar também no sys.modules
        import moviasai.covariates
        sys.modules['covariates'] = moviasai.covariates
        
        logger.debug("✅ Encoders customizados registrados")
    
    def _load_model(self, model_name: str, category: str, segment: int):
        """
        Carrega modelo do disco (com cache)
        
        Carrega de: {models_root_dir}/{model_name}/dataset_{category}_*_class{segment}_*/model_00.pkl
        
        Parameters
        ----------
        model_name : str
            Nome do modelo (ex: 'DLinearModel', 'TSMixerModel')
        category : str
            Categoria ('km' ou 'h')
        segment : int
            Número do segmento (0, 1, 2)
        
        Returns
        -------
        model
            Modelo Darts carregado
        
        Raises
        ------
        FileNotFoundError
            Se modelo não for encontrado
        ValueError
            Se classe do modelo não estiver mapeada
        """
        cache_key = (category, segment, model_name)
        
        if cache_key in self._model_cache:
            logger.debug(f"Modelo em cache: {cache_key}")
            return self._model_cache[cache_key]
        
        # Encontrar diretório do modelo
        model_dir = self.config.find_model_path(model_name, category, segment)
        
        if model_dir is None:
            raise FileNotFoundError(
                f"Diretório do modelo não encontrado: {model_name} "
                f"(category={category}, segment={segment})"
            )
        
        # Path do arquivo do modelo
        model_file = model_dir / "model_00.pkl"
        
        if not model_file.exists():
            raise FileNotFoundError(
                f"Arquivo do modelo não encontrado: {model_file}"
            )
        
        # Obter classe do modelo
        if model_name not in MODEL_CLASS_MAP:
            available = ', '.join(MODEL_CLASS_MAP.keys())
            raise ValueError(
                f"Modelo não suportado: {model_name}. "
                f"Modelos disponíveis: {available}"
            )
        
        model_class = MODEL_CLASS_MAP[model_name]
        
        logger.info(f"Carregando modelo: {model_file}")
        
        # Carregar modelo usando método .load() da classe
        try:
            model = model_class.load(str(model_file))
        except Exception as e:
            logger.error(f"Erro ao carregar modelo {model_file}: {e}")
            raise
        
        # Cache
        self._model_cache[cache_key] = model
        
        logger.info(
            f"✅ Modelo carregado: {model_name} "
            f"(category={category}, segment={segment})"
        )
        
        return model
    
    def _get_predictor(self, category: str, segment: int) -> Predictor:
        """
        Obtém predictor para categoria/segmento (com cache)
        
        Parameters
        ----------
        category : str
            Categoria ('km' ou 'h')
        segment : int
            Número do segmento
        
        Returns
        -------
        Predictor
            Predictor configurado para o segmento
        
        Raises
        ------
        ValueError
            Se configuração não for encontrada
        FileNotFoundError
            Se modelo não for encontrado
        """
        cache_key = (category, segment)
        
        if cache_key in self._predictor_cache:
            logger.debug(f"Predictor em cache: {cache_key}")
            return self._predictor_cache[cache_key]
        
        # Obter configuração do segmento
        seg_config = self.config.get_segment_config(category, segment)
        
        if seg_config is None:
            raise ValueError(
                f"Configuração não encontrada para category={category}, segment={segment}"
            )
        
        # Carregar modelos
        models = []
        weights = []
        
        for model_config in seg_config.models:
            model = self._load_model(
                model_config.model_name,
                category,
                segment
            )
            models.append(model)
            weights.append(model_config.weight)
        
        # Criar predictor usando configurações globais
        if len(models) == 1:
            predictor = Predictor(
                model=models[0],
                history_size=self.config.history_size,
                forecast_horizon=self.config.forecast_horizon,
                max_steps=self.config.max_steps,
                verbose=False
            )
            logger.info(
                f"✅ Predictor criado (modelo único): "
                f"category={category}, segment={segment}, "
                f"model={seg_config.models[0].model_name}"
            )
        else:
            predictor = Predictor(
                model=models,
                model_weights=weights,
                history_size=self.config.history_size,
                forecast_horizon=self.config.forecast_horizon,
                max_steps=self.config.max_steps,
                verbose=False
            )
            logger.info(
                f"✅ Predictor criado (ensemble): "
                f"category={category}, segment={segment}, "
                f"models={[m.model_name for m in seg_config.models]}, "
                f"weights={weights}"
            )
        
        self._predictor_cache[cache_key] = predictor
        
        return predictor
    
    def _create_timeseries_from_profile(
        self,
        profile: Dict,
        history_size: int
    ) -> Tuple[TimeSeries, TimeSeries]:
        """
        Cria TimeSeries a partir do perfil do veículo
        
        Segue o mesmo padrão usado no treinamento:
        1. Normalizar valores usando upper (percentil 99)
        2. Static covariates (15 summary features)
        3. TimeSeries principal com valores normalizados
        4. Daily features (10 features × 7 dias = 70 features)
        
        Parameters
        ----------
        profile : Dict
            Perfil completo do veículo (85 features + metadata)
        history_size : int
            Tamanho do histórico (últimos N dias)
        
        Returns
        -------
        tuple
            (ts, ts_daily) onde:
            - ts: TimeSeries principal (valores normalizados) com static covariates
            - ts_daily: TimeSeries de daily features (past covariates)
        
        Raises
        ------
        ValueError
            Se veículo não tiver samples suficientes
        """
        # Validar samples
        samples = profile['samples']
        if len(samples) < history_size:
            raise ValueError(
                f"Veículo {profile['vehicle_id']}: dados insuficientes. "
                f"Necessário: {history_size}, disponível: {len(samples)}"
            )
        
        # Extrair últimos history_size valores
        values_raw = np.array(samples[-history_size:])
        
        # 1. NORMALIZAR (igual ao treinamento)
        upper = profile['upper']
        values_clipped = np.clip(values_raw, 0, upper)
        max_value = max(values_clipped)
        values_scaled = (
            values_clipped / max_value if max_value > 0 else np.zeros_like(values_clipped)
        ).reshape(-1, 1)
        
        # Criar índice temporal (últimos history_size dias)
        last_date = pd.Timestamp(profile['samples_end_date'])
        time_index = pd.date_range(
            end=last_date,
            periods=history_size,
            freq='D'
        )
        
        # 2. STATIC COVARIATES (15 summary features)
        category = profile['category']
        
        segmentation_features = [
            f'{category}_por_dia',
            f'media_{category}',
            f'max_{category}',
            f'mediana_{category}',
            f'std_{category}',
            f'score_continuidade_{category}',
            f'taxa_semanas_ativas_{category}',
            f'taxa_dias_ativos_{category}',
            f'cv_gaps_{category}',
            f'gap_medio_{category}',
            f'gap_max_{category}',
            f'cv_{category}',
            f'p25_{category}',
            f'p75_{category}',
            f'iqr_{category}'
        ]
        
        static_cov_dict = {feat: profile[feat] for feat in segmentation_features}
        static_cov = pd.DataFrame([static_cov_dict])
        
        # 3. TIMESERIES principal
        target_col = f'{category}_dia_clean'
        
        ts = TimeSeries.from_times_and_values(
            times=time_index,
            values=values_scaled,
            static_covariates=static_cov,
            freq='D',
            fill_missing_dates=False,
            columns=[f'{target_col}_scaled']
        )
        
        # 4. DAILY FEATURES (10 features por dia da semana)
        weekday_base_features = [
            'mean', 'std', 'median', 'max', 'min',
            'p25', 'p75', 'iqr', 'prob_active', 'cv'
        ]
        
        n_dates = len(time_index)
        n_features = len(weekday_base_features)
        daily_features_matrix = np.zeros((n_dates, n_features))
        
        # Obter dia da semana (1=segunda, 7=domingo)
        weekdays = [date.weekday() + 1 for date in time_index]
        
        for i, weekday in enumerate(weekdays):
            for j, feat in enumerate(weekday_base_features):
                col_name = f'day_{weekday}_{feat}'
                daily_features_matrix[i, j] = profile[col_name]
        
        # 5. TIMESERIES DAILY (past covariates)
        ts_daily = TimeSeries.from_times_and_values(
            times=time_index,
            values=daily_features_matrix,
            freq='D',
            fill_missing_dates=False,
            columns=weekday_base_features
        )
        
        return ts, ts_daily
    
    async def predict_date_to_reach(
        self,
        vehicle_ids: List[int],
        target_values: List[float],
        n_jobs: int = 1
    ) -> List[Dict]:
        """
        Prediz data para atingir valor acumulado
        
        Encontra a PRIMEIRA data em que o valor acumulado DAS PREDIÇÕES
        ultrapassa o target_value. O acumulado é calculado APENAS sobre as
        predições futuras, começando do zero.
        
        Parameters
        ----------
        vehicle_ids : List[int]
            Lista de IDs de veículos
        target_values : List[float]
            Valores alvo (acumulado) para cada veículo
        n_jobs : int, default=1
            Número de threads para paralelização
            - 1: sequencial
            - -1: usar todos os CPUs disponíveis
            - n > 1: usar n threads
        
        Returns
        -------
        List[Dict]
            Lista com resultados:
            - vehicle_id: int
            - category: str
            - segment: int
            - target_value: float
            - predicted_date: str (ISO format)
            - n_steps: int (dias até atingir)
            - accumulated_value: float (valor acumulado previsto)
        
        Raises
        ------
        ValueError
            Se tamanhos de vehicle_ids e target_values não coincidirem
        FileNotFoundError
            Se modelo não for encontrado
        
        Examples
        --------
        >>> results = await service.predict_date_to_reach(
        ...     vehicle_ids=[1316, 18230],
        ...     target_values=[10000.0, 15000.0],
        ...     n_jobs=-1
        ... )
        >>> print(f"Veículo 1316 atingirá 10.000 km em {results[0]['predicted_date']}")
        """
        if len(vehicle_ids) != len(target_values):
            raise ValueError(
                f"vehicle_ids e target_values devem ter mesmo tamanho "
                f"({len(vehicle_ids)} != {len(target_values)})"
            )
        
        logger.info(f"Iniciando predição date-to-reach para {len(vehicle_ids)} veículos")
        
        # Buscar perfis
        profiles = await self.profile_service.get_profiles_batch(vehicle_ids)
        
        if not profiles:
            logger.warning("Nenhum perfil encontrado")
            return []
        
        # Agrupar por (category, segment)
        groups: Dict[Tuple[str, int], List[Tuple[int, Dict, float]]] = {}
        
        for vid, target in zip(vehicle_ids, target_values):
            if vid not in profiles:
                logger.warning(f"Perfil não encontrado para veículo {vid}")
                continue
            
            profile = profiles[vid]
            key = (profile['category'], profile['segment'])
            
            if key not in groups:
                groups[key] = []
            
            groups[key].append((vid, profile, target))
        
        # Processar cada grupo
        all_results = []
        
        for (category, segment), group_data in groups.items():
            logger.info(
                f"Processando grupo: category={category}, segment={segment}, "
                f"n_vehicles={len(group_data)}"
            )
            
            # Obter predictor
            try:
                predictor = self._get_predictor(category, segment)
            except (ValueError, FileNotFoundError) as e:
                logger.error(f"Erro ao obter predictor para {category}-{segment}: {e}")
                continue
            
            history_size = predictor.history_size
            
            # Criar TimeSeries para cada veículo do grupo
            histories = []
            past_covs = []
            targets = []
            vehicle_ids_group = []
            
            for vid, profile, target in group_data:
                try:
                    ts, ts_daily = self._create_timeseries_from_profile(
                        profile,
                        history_size
                    )
                    
                    histories.append(ts)
                    past_covs.append(ts_daily)
                    targets.append(target)
                    vehicle_ids_group.append(vid)
                    
                except ValueError as e:
                    logger.warning(f"Veículo {vid}: {e}")
                    continue
            
            if not histories:
                logger.warning(f"Nenhuma série válida para category={category}, segment={segment}")
                continue
            
            logger.info(f"Predizendo {len(histories)} séries com n_jobs={n_jobs}")
            
            # Predizer em lote
            result = predictor.date_to_reach_accumulated(
                history=histories,
                target_value=targets,
                past_covariates=past_covs if predictor.supports_past_covariates else None,
                n_jobs=n_jobs
            )
            
            # Formatar resultados
            dates = result['date'] if isinstance(result['date'], list) else [result['date']]
            n_steps_list = result['n_steps'] if isinstance(result['n_steps'], list) else [result['n_steps']]
            paths = result['path'] if isinstance(result['path'], list) else [result['path']]
            
            for vid, date, n_steps, path in zip(vehicle_ids_group, dates, n_steps_list, paths):
                accumulated = float(path.values()[:, 0].sum())
                target_idx = vehicle_ids_group.index(vid)
                
                all_results.append({
                    'vehicle_id': vid,
                    'category': category,
                    'segment': segment,
                    'target_value': targets[target_idx],
                    'predicted_date': date.isoformat(),
                    'n_steps': n_steps,
                    'accumulated_value': accumulated
                })
            
            logger.info(f"✅ Grupo processado: {len(all_results)} resultados")
        
        logger.info(f"Predição date-to-reach concluída: {len(all_results)} resultados")
        
        return all_results
    
    async def predict_accumulated_at_step(
        self,
        vehicle_ids: List[int],
        n_steps: Optional[List[int]] = None,
        reference_dates: Optional[List[str]] = None,
        n_jobs: int = 1
    ) -> List[Dict]:
        """
        Prediz valor acumulado em uma data/step específico
        
        Calcula o valor acumulado DAS PREDIÇÕES em uma data de referência
        ou após n_steps. O acumulado é calculado APENAS sobre as predições
        futuras, começando do zero.
        
        Parameters
        ----------
        vehicle_ids : List[int]
            Lista de IDs de veículos
        n_steps : List[int], optional
            Número de steps para cada veículo
        reference_dates : List[str], optional
            Datas de referência (ISO format) para cada veículo
        n_jobs : int, default=1
            Número de threads para paralelização
        
        Returns
        -------
        List[Dict]
            Lista com resultados:
            - vehicle_id: int
            - category: str
            - segment: int
            - n_steps: int
            - reference_date: str (ISO format)
            - accumulated_value: float
        
        Raises
        ------
        ValueError
            Se nem n_steps nem reference_dates forem fornecidos,
            ou se tamanhos não coincidirem
        
        Examples
        --------
        >>> # Quanto vai acumular em 90 dias?
        >>> results = await service.predict_accumulated_at_step(
        ...     vehicle_ids=[1316, 18230],
        ...     n_steps=[90, 90],
        ...     n_jobs=-1
        ... )
        
        >>> # Quanto vai acumular até 31/12/2024?
        >>> results = await service.predict_accumulated_at_step(
        ...     vehicle_ids=[1316],
        ...     reference_dates=["2024-12-31"]
        ... )
        """
        if n_steps is None and reference_dates is None:
            raise ValueError("Deve fornecer n_steps ou reference_dates")
        
        n_vehicles = len(vehicle_ids)
        
        logger.info(f"Iniciando predição accumulated-at-step para {n_vehicles} veículos")
        
        # Normalizar n_steps
        if n_steps is None:
            n_steps_list = [None] * n_vehicles
        elif isinstance(n_steps, int):
            n_steps_list = [n_steps] * n_vehicles
        else:
            n_steps_list = n_steps
            if len(n_steps_list) != n_vehicles:
                raise ValueError(
                    f"n_steps deve ter mesmo tamanho que vehicle_ids "
                    f"({len(n_steps_list)} != {n_vehicles})"
                )
        
        # Normalizar reference_dates
        if reference_dates is None:
            ref_dates_list = [None] * n_vehicles
        elif isinstance(reference_dates, str):
            ref_dates_list = [pd.Timestamp(reference_dates)] * n_vehicles
        else:
            ref_dates_list = [pd.Timestamp(d) if d else None for d in reference_dates]
            if len(ref_dates_list) != n_vehicles:
                raise ValueError(
                    f"reference_dates deve ter mesmo tamanho que vehicle_ids "
                    f"({len(ref_dates_list)} != {n_vehicles})"
                )
        
        # Buscar perfis
        profiles = await self.profile_service.get_profiles_batch(vehicle_ids)
        
        if not profiles:
            logger.warning("Nenhum perfil encontrado")
            return []
        
        # Agrupar por (category, segment)
        groups: Dict[Tuple[str, int], List] = {}
        
        for i, vid in enumerate(vehicle_ids):
            if vid not in profiles:
                logger.warning(f"Perfil não encontrado para veículo {vid}")
                continue
            
            profile = profiles[vid]
            key = (profile['category'], profile['segment'])
            
            if key not in groups:
                groups[key] = []
            
            groups[key].append((vid, profile, n_steps_list[i], ref_dates_list[i]))
        
        # Processar cada grupo
        all_results = []
        
        for (category, segment), group_data in groups.items():
            logger.info(
                f"Processando grupo: category={category}, segment={segment}, "
                f"n_vehicles={len(group_data)}"
            )
            
            # Obter predictor
            try:
                predictor = self._get_predictor(category, segment)
            except (ValueError, FileNotFoundError) as e:
                logger.error(f"Erro ao obter predictor para {category}-{segment}: {e}")
                continue
            
            history_size = predictor.history_size
            
            # Criar TimeSeries
            histories = []
            past_covs = []
            n_steps_group = []
            ref_dates_group = []
            vehicle_ids_group = []
            
            for vid, profile, n_step, ref_date in group_data:
                try:
                    ts, ts_daily = self._create_timeseries_from_profile(
                        profile,
                        history_size
                    )
                    
                    histories.append(ts)
                    past_covs.append(ts_daily)
                    n_steps_group.append(n_step)
                    ref_dates_group.append(ref_date)
                    vehicle_ids_group.append(vid)
                    
                except ValueError as e:
                    logger.warning(f"Veículo {vid}: {e}")
                    continue
            
            if not histories:
                logger.warning(f"Nenhuma série válida para category={category}, segment={segment}")
                continue
            
            logger.info(f"Predizendo {len(histories)} séries com n_jobs={n_jobs}")
            
            # Predizer em lote
            result = predictor.predict_accumulated_at_step(
                history=histories,
                n_steps=n_steps_group if any(s is not None for s in n_steps_group) else None,
                reference_date=ref_dates_group if any(d is not None for d in ref_dates_group) else None,
                past_covariates=past_covs if predictor.supports_past_covariates else None,
                n_jobs=n_jobs
            )
            
            # Formatar resultados
            accumulated_values = (
                result['accumulated_value'] 
                if isinstance(result['accumulated_value'], list) 
                else [result['accumulated_value']]
            )
            paths = result['path'] if isinstance(result['path'], list) else [result['path']]
            
            for i, (vid, acc_value, path) in enumerate(zip(vehicle_ids_group, accumulated_values, paths)):
                # Determinar n_steps e reference_date finais
                final_n_steps = n_steps_group[i] if n_steps_group[i] is not None else len(path)
                
                if ref_dates_group[i] is not None:
                    final_ref_date = ref_dates_group[i]
                else:
                    last_date = pd.Timestamp(profiles[vid]['samples_end_date'])
                    final_ref_date = last_date + pd.Timedelta(days=final_n_steps)
                
                all_results.append({
                    'vehicle_id': vid,
                    'category': category,
                    'segment': segment,
                    'n_steps': final_n_steps,
                    'reference_date': final_ref_date.isoformat(),
                    'accumulated_value': float(acc_value)
                })
            
            logger.info(f"✅ Grupo processado: {len(all_results)} resultados")
        
        logger.info(f"Predição accumulated-at-step concluída: {len(all_results)} resultados")
        
        return all_results
    
    def get_cache_stats(self) -> Dict[str, int]:
        """
        Retorna estatísticas do cache
        
        Returns
        -------
        dict
            - models_loaded: número de modelos em cache
            - predictors_cached: número de predictors em cache
        """
        return {
            'models_loaded': len(self._model_cache),
            'predictors_cached': len(self._predictor_cache)
        }
    
    def clear_cache(self):
        """Limpa cache de modelos e predictors"""
        self._model_cache.clear()
        self._predictor_cache.clear()
        logger.info("Cache limpo")
    
    def warmup_cache(self):
        """
        Pré-carrega todos os modelos configurados
        
        Útil para inicialização do serviço.
        """
        logger.info("Iniciando warmup do cache...")
        
        loaded = []
        failed = []
        
        for cat_config in self.config.categories:
            for seg_config in cat_config.segments:
                try:
                    # Criar predictor (que carrega os modelos)
                    predictor = self._get_predictor(
                        cat_config.category,
                        seg_config.segment
                    )
                    
                    loaded.append(f"{cat_config.category}_{seg_config.segment}")
                    
                except Exception as e:
                    failed.append(f"{cat_config.category}_{seg_config.segment}: {e}")
                    logger.error(
                        f"Erro ao carregar predictor {cat_config.category}_{seg_config.segment}: {e}"
                    )
        
        logger.info(f"✅ Warmup concluído: {len(loaded)} predictors carregados")
        
        if failed:
            logger.warning(f"⚠️  {len(failed)} predictors falharam")
        
        return {'loaded': loaded, 'failed': failed}

    async def predict_date_to_reach_with_profiles(
        self,
        vehicle_ids: List[int],
        profiles: Dict[int, Dict],
        target_values: List[float],
        n_jobs: int = 1
    ) -> List[Dict]:
        """
        Versão interna que aceita perfis customizados
        
        Usado pelo EvaluationService para evitar buscar do banco
        """
        if len(vehicle_ids) != len(target_values):
            raise ValueError(
                f"vehicle_ids e target_values devem ter mesmo tamanho "
                f"({len(vehicle_ids)} != {len(target_values)})"
            )
        
        logger.info(f"Iniciando predição date-to-reach para {len(vehicle_ids)} veículos (com perfis customizados)")
        
        # Agrupar por (category, segment)
        groups: Dict[Tuple[str, int], List[Tuple[int, Dict, float]]] = {}
        
        for vid, target in zip(vehicle_ids, target_values):
            if vid not in profiles:
                logger.warning(f"Perfil não encontrado para veículo {vid}")
                continue
            
            profile = profiles[vid]
            key = (profile['category'], profile['segment'])
            
            if key not in groups:
                groups[key] = []
            
            groups[key].append((vid, profile, target))
        
        # Processar cada grupo (mesmo código do método original)
        all_results = []
        
        for (category, segment), group_data in groups.items():
            logger.info(
                f"Processando grupo: category={category}, segment={segment}, "
                f"n_vehicles={len(group_data)}"
            )
            
            try:
                predictor = self._get_predictor(category, segment)
            except (ValueError, FileNotFoundError) as e:
                logger.error(f"Erro ao obter predictor para {category}-{segment}: {e}")
                continue
            
            history_size = predictor.history_size
            
            # Criar TimeSeries
            histories = []
            past_covs = []
            targets = []
            vehicle_ids_group = []
            
            for vid, profile, target in group_data:
                try:
                    ts, ts_daily = self._create_timeseries_from_profile(
                        profile,
                        history_size
                    )
                    
                    histories.append(ts)
                    past_covs.append(ts_daily)
                    targets.append(target)
                    vehicle_ids_group.append(vid)
                    
                except ValueError as e:
                    logger.warning(f"Veículo {vid}: {e}")
                    continue
            
            if not histories:
                logger.warning(f"Nenhuma série válida para category={category}, segment={segment}")
                continue
            
            logger.info(f"Predizendo {len(histories)} séries com n_jobs={n_jobs}")
            
            # Predizer em lote
            result = predictor.date_to_reach_accumulated(
                history=histories,
                target_value=targets,
                past_covariates=past_covs if predictor.supports_past_covariates else None,
                n_jobs=n_jobs
            )
            
            # Formatar resultados
            dates = result['date'] if isinstance(result['date'], list) else [result['date']]
            n_steps_list = result['n_steps'] if isinstance(result['n_steps'], list) else [result['n_steps']]
            paths = result['path'] if isinstance(result['path'], list) else [result['path']]
            
            for vid, date, n_steps, path in zip(vehicle_ids_group, dates, n_steps_list, paths):
                accumulated = float(path.values()[:, 0].sum())
                target_idx = vehicle_ids_group.index(vid)
                
                all_results.append({
                    'vehicle_id': vid,
                    'category': category,
                    'segment': segment,
                    'target_value': targets[target_idx],
                    'predicted_date': date.isoformat(),
                    'n_steps': n_steps,
                    'accumulated_value': accumulated
                })
            
            logger.info(f"✅ Grupo processado: {len(all_results)} resultados")
        
        logger.info(f"Predição date-to-reach concluída: {len(all_results)} resultados")
        
        return all_results


    async def predict_accumulated_at_step_with_profiles(
        self,
        vehicle_ids: List[int],
        profiles: Dict[int, Dict],
        n_steps: Optional[List[int]] = None,
        reference_dates: Optional[List[str]] = None,
        n_jobs: int = 1
    ) -> List[Dict]:
        """
        Versão interna que aceita perfis customizados
        
        Usado pelo EvaluationService para evitar buscar do banco
        """
        if n_steps is None and reference_dates is None:
            raise ValueError("Deve fornecer n_steps ou reference_dates")
        
        n_vehicles = len(vehicle_ids)
        
        logger.info(f"Iniciando predição accumulated-at-step para {n_vehicles} veículos (com perfis customizados)")
        
        # Normalizar n_steps
        if n_steps is None:
            n_steps_list = [None] * n_vehicles
        elif isinstance(n_steps, int):
            n_steps_list = [n_steps] * n_vehicles
        else:
            n_steps_list = n_steps
            if len(n_steps_list) != n_vehicles:
                raise ValueError(
                    f"n_steps deve ter mesmo tamanho que vehicle_ids "
                    f"({len(n_steps_list)} != {n_vehicles})"
                )
        
        # Normalizar reference_dates
        if reference_dates is None:
            ref_dates_list = [None] * n_vehicles
        elif isinstance(reference_dates, str):
            ref_dates_list = [pd.Timestamp(reference_dates)] * n_vehicles
        else:
            ref_dates_list = [pd.Timestamp(d) if d else None for d in reference_dates]
            if len(ref_dates_list) != n_vehicles:
                raise ValueError(
                    f"reference_dates deve ter mesmo tamanho que vehicle_ids "
                    f"({len(ref_dates_list)} != {n_vehicles})"
                )
        
        # Agrupar por (category, segment)
        groups: Dict[Tuple[str, int], List] = {}
        
        for i, vid in enumerate(vehicle_ids):
            if vid not in profiles:
                logger.warning(f"Perfil não encontrado para veículo {vid}")
                continue
            
            profile = profiles[vid]
            key = (profile['category'], profile['segment'])
            
            if key not in groups:
                groups[key] = []
            
            groups[key].append((vid, profile, n_steps_list[i], ref_dates_list[i]))
        
        # Processar cada grupo (mesmo código do método original)
        all_results = []
        
        for (category, segment), group_data in groups.items():
            logger.info(
                f"Processando grupo: category={category}, segment={segment}, "
                f"n_vehicles={len(group_data)}"
            )
            
            try:
                predictor = self._get_predictor(category, segment)
            except (ValueError, FileNotFoundError) as e:
                logger.error(f"Erro ao obter predictor para {category}-{segment}: {e}")
                continue
            
            history_size = predictor.history_size
            
            # Criar TimeSeries
            histories = []
            past_covs = []
            n_steps_group = []
            ref_dates_group = []
            vehicle_ids_group = []
            
            for vid, profile, n_step, ref_date in group_data:
                try:
                    ts, ts_daily = self._create_timeseries_from_profile(
                        profile,
                        history_size
                    )
                    
                    histories.append(ts)
                    past_covs.append(ts_daily)
                    n_steps_group.append(n_step)
                    ref_dates_group.append(ref_date)
                    vehicle_ids_group.append(vid)
                    
                except ValueError as e:
                    logger.warning(f"Veículo {vid}: {e}")
                    continue
            
            if not histories:
                logger.warning(f"Nenhuma série válida para category={category}, segment={segment}")
                continue
            
            logger.info(f"Predizendo {len(histories)} séries com n_jobs={n_jobs}")
            
            # Predizer em lote
            result = predictor.predict_accumulated_at_step(
                history=histories,
                n_steps=n_steps_group if any(s is not None for s in n_steps_group) else None,
                reference_date=ref_dates_group if any(d is not None for d in ref_dates_group) else None,
                past_covariates=past_covs if predictor.supports_past_covariates else None,
                n_jobs=n_jobs
            )
            
            # Formatar resultados
            accumulated_values = (
                result['accumulated_value'] 
                if isinstance(result['accumulated_value'], list) 
                else [result['accumulated_value']]
            )
            paths = result['path'] if isinstance(result['path'], list) else [result['path']]
            
            for i, (vid, acc_value, path) in enumerate(zip(vehicle_ids_group, accumulated_values, paths)):
                final_n_steps = n_steps_group[i] if n_steps_group[i] is not None else len(path)
                
                if ref_dates_group[i] is not None:
                    final_ref_date = ref_dates_group[i]
                else:
                    last_date = pd.Timestamp(profiles[vid]['samples_end_date'])
                    final_ref_date = last_date + pd.Timedelta(days=final_n_steps)
                
                all_results.append({
                    'vehicle_id': vid,
                    'category': category,
                    'segment': segment,
                    'n_steps': final_n_steps,
                    'reference_date': final_ref_date.isoformat(),
                    'accumulated_value': float(acc_value)
                })
            
            logger.info(f"✅ Grupo processado: {len(all_results)} resultados")
        
        logger.info(f"Predição accumulated-at-step concluída: {len(all_results)} resultados")
        
        return all_results
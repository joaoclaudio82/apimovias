import numpy as np
import pandas as pd
from typing import Union, List, Optional, Tuple, Dict
from darts import TimeSeries
from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
import warnings
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp


class Predictor:
    """
    Preditor para manutenção preventiva com suporte a paralelização
    
    Características:
    - Suporta modelo único ou ensemble
    - Predição autoregressiva pontual
    - Paralelização para múltiplas séries
    - Acumulado calculado APENAS sobre predições (não inclui histórico)
    
    Examples
    --------
    # Modelo único
    >>> predictor = Predictor(model=model, history_size=28, forecast_horizon=7)
    
    # Ensemble
    >>> predictor = Predictor(
    ...     model=[model1, model2, model3],
    ...     model_weights=[0.5, 0.3, 0.2],
    ...     history_size=28,
    ...     forecast_horizon=7
    ... )
    
    # Encontrar quando vai acumular 10000 km (a partir de agora)
    >>> result = predictor.date_to_reach_accumulated(
    ...     history=ts,
    ...     target_value=10000.0
    ... )
    >>> print(f"Acumulará 10000 km em {result['date']} ({result['n_steps']} dias)")
    
    # Quanto vai acumular em 90 dias?
    >>> result = predictor.predict_accumulated_at_step(
    ...     history=ts,
    ...     n_steps=90
    ... )
    >>> print(f"Acumulará {result['accumulated_value']:.2f} km em 90 dias")
    """
    
    def __init__(
        self,
        model: Union[
            TorchForecastingModel,
            GlobalForecastingModel,
            List[Union[TorchForecastingModel, GlobalForecastingModel]]
        ],
        history_size: int = 28,
        forecast_horizon: int = 7,
        max_steps: int = 365,
        model_weights: Optional[List[float]] = None,
        verbose: bool = False
    ):
        """
        Inicializa o preditor
        
        Parameters
        ----------
        model : modelo único ou lista de modelos
            TorchForecastingModel, GlobalForecastingModel ou lista deles
        history_size : int, default=28
            Tamanho do histórico usado no treinamento
        forecast_horizon : int, default=7
            Horizonte de previsão usado no treinamento
        max_steps : int, default=365
            Número máximo de steps para predição autoregressiva
        model_weights : List[float], optional
            Pesos para cada modelo (deve somar 1.0).
            Obrigatório se model for lista.
        verbose : bool, default=False
            Se True, imprime informações detalhadas
        """
        self.history_size = history_size
        self.forecast_horizon = forecast_horizon
        self.max_steps = max_steps
        self.verbose = verbose
        
        # Normalizar model para lista
        if isinstance(model, list):
            self.models = model
            self.is_ensemble = len(model) > 1
        else:
            self.models = [model]
            self.is_ensemble = False
        
        # Validar e configurar pesos
        if model_weights is not None:
            if len(model_weights) != len(self.models):
                raise ValueError(
                    f"model_weights deve ter mesmo tamanho que model "
                    f"({len(model_weights)} != {len(self.models)})"
                )
            if not np.isclose(sum(model_weights), 1.0):
                raise ValueError(
                    f"Soma dos model_weights deve ser 1.0, recebido: {sum(model_weights)}"
                )
            self.model_weights = model_weights
        else:
            if self.is_ensemble:
                raise ValueError("model_weights é obrigatório quando model é uma lista")
            self.model_weights = [1.0]
        
        # Validar capacidades dos modelos
        first_model = self.models[0]
        self.supports_past_covariates = first_model.supports_past_covariates
        self.supports_future_covariates = first_model.supports_future_covariates
        
        for m in self.models[1:]:
            if m.supports_past_covariates != self.supports_past_covariates:
                raise ValueError(
                    "Todos os modelos devem ter mesmo suporte para past_covariates"
                )
            if m.supports_future_covariates != self.supports_future_covariates:
                raise ValueError(
                    "Todos os modelos devem ter mesmo suporte para future_covariates"
                )
        
        if self.verbose:
            print(f"✓ Predictor inicializado")
            if self.is_ensemble:
                print(f"  Ensemble: {len(self.models)} modelos")
                print(f"  Pesos: {self.model_weights}")
            else:
                print(f"  Modelo: {self.models[0].__class__.__name__}")
            print(f"  History: {history_size}, Horizon: {forecast_horizon}")
            print(f"  Past covariates: {self.supports_past_covariates}")
            print(f"  Future covariates: {self.supports_future_covariates}")
    
    def _normalize_inputs(
        self,
        history: Union[TimeSeries, List[TimeSeries]],
        target_value: Union[float, List[float]],
        past_covariates: Optional[Union[TimeSeries, List[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, List[TimeSeries]]] = None
    ) -> Tuple[List[TimeSeries], List[float], Optional[List[TimeSeries]], Optional[List[TimeSeries]], bool]:
        """Normaliza entradas para listas e retorna se é entrada única"""
        # History
        if isinstance(history, TimeSeries):
            history_list = [history]
            is_single = True
        else:
            history_list = history
            is_single = False
        
        n_series = len(history_list)
        
        # Target value
        if isinstance(target_value, (int, float)):
            if is_single:
                target_list = [float(target_value)]
            else:
                # Mesmo valor para todas as séries
                target_list = [float(target_value)] * n_series
        else:
            target_list = [float(v) for v in target_value]
            if len(target_list) != n_series:
                raise ValueError(
                    f"target_value deve ter mesmo comprimento que history "
                    f"({len(target_list)} != {n_series})"
                )
        
        # Past covariates
        if past_covariates is None:
            past_cov_list = None
        elif isinstance(past_covariates, TimeSeries):
            if not is_single:
                raise ValueError(
                    "Se history é lista, past_covariates deve ser lista ou None"
                )
            past_cov_list = [past_covariates]
        else:
            past_cov_list = past_covariates
            if len(past_cov_list) != n_series:
                raise ValueError(
                    f"past_covariates deve ter mesmo comprimento que history "
                    f"({len(past_cov_list)} != {n_series})"
                )
        
        # Future covariates
        if future_covariates is None:
            future_cov_list = None
        elif isinstance(future_covariates, TimeSeries):
            if not is_single:
                raise ValueError(
                    "Se history é lista, future_covariates deve ser lista ou None"
                )
            future_cov_list = [future_covariates]
        else:
            future_cov_list = future_covariates
            if len(future_cov_list) != n_series:
                raise ValueError(
                    f"future_covariates deve ter mesmo comprimento que history "
                    f"({len(future_cov_list)} != {n_series})"
                )
        
        return history_list, target_list, past_cov_list, future_cov_list, is_single
    
    def _get_accumulated_value(self, ts: TimeSeries) -> float:
        """Calcula valor acumulado de uma série"""
        return float(ts.values()[:, 0].sum())
    
    def _get_last_history_value(self, ts: TimeSeries) -> float:
        """Obtém último valor do histórico"""
        return float(ts.values()[-1, 0])
    
    def _extend_covariates(
        self,
        covariates: Optional[TimeSeries],
        target_length: int
    ) -> Optional[TimeSeries]:
        """Estende covariáveis usando forward fill"""
        if covariates is None:
            return None
        
        current_length = len(covariates)
        
        if current_length >= target_length:
            return covariates
        
        n_extend = target_length - current_length
        
        # Repetir último valor
        last_values = covariates.values()[-1:, :]
        extended_values = np.repeat(last_values, n_extend, axis=0)
        
        # Criar nova série
        last_date = covariates.end_time()
        freq = covariates.freq
        
        extended_dates = pd.date_range(
            start=last_date + freq,
            periods=n_extend,
            freq=freq
        )
        
        extended_ts = TimeSeries.from_times_and_values(
            times=extended_dates,
            values=extended_values,
            columns=covariates.components
        )
        
        return covariates.append(extended_ts)
    
    def _predict_single_model(
        self,
        model,
        n: int,
        series: TimeSeries,
        past_covariates: Optional[TimeSeries] = None,
        future_covariates: Optional[TimeSeries] = None
    ) -> TimeSeries:
        """Predição com um único modelo"""
        return model.predict(
            n=n,
            series=series,
            past_covariates=past_covariates if self.supports_past_covariates else None,
            future_covariates=future_covariates if self.supports_future_covariates else None
        )
    
    def _ensemble_predict(
        self,
        n: int,
        series: TimeSeries,
        past_covariates: Optional[TimeSeries] = None,
        future_covariates: Optional[TimeSeries] = None
    ) -> TimeSeries:
        """Predição com ensemble (média ponderada)"""
        predictions = []
        
        for model in self.models:
            pred = self._predict_single_model(
                model, n, series, past_covariates, future_covariates
            )
            predictions.append(pred.values())
        
        # Média ponderada
        pred_arrays = np.array(predictions)  # (n_models, n_steps, 1)
        weighted_pred = np.average(pred_arrays, axis=0, weights=self.model_weights)
        
        # Criar TimeSeries
        first_pred = self._predict_single_model(self.models[0], n, series, past_covariates, future_covariates)
        
        return TimeSeries.from_times_and_values(
            times=first_pred.time_index,
            values=weighted_pred
        )
    
    def _autoregressive_predict(
        self,
        history: TimeSeries,
        n_steps: int,
        past_covariates: Optional[TimeSeries] = None,
        future_covariates: Optional[TimeSeries] = None
    ) -> TimeSeries:
        """
        Predição autoregressiva para n_steps > forecast_horizon
        
        Returns
        -------
        TimeSeries
            Série temporal com n_steps predições
        """
        if n_steps <= self.forecast_horizon:
            # Predição direta
            if self.is_ensemble:
                return self._ensemble_predict(
                    n_steps, history, past_covariates, future_covariates
                )
            else:
                return self._predict_single_model(
                    self.models[0], n_steps, history, past_covariates, future_covariates
                )
        
        # Predição autoregressiva
        all_predictions = []
        current_history = history
        current_past_cov = past_covariates
        
        # Estender future_covariates se necessário
        if future_covariates is not None and self.supports_future_covariates:
            future_covariates = self._extend_covariates(
                future_covariates,
                len(history) + n_steps
            )
        
        steps_remaining = n_steps
        step_offset = 0
        
        while steps_remaining > 0:
            # Predizer próximo horizonte
            n_predict = min(self.forecast_horizon, steps_remaining)
            
            # Preparar covariáveis para esta iteração
            iter_future_cov = None
            if future_covariates is not None and self.supports_future_covariates:
                start_idx = len(current_history)
                end_idx = start_idx + n_predict
                iter_future_cov = future_covariates[start_idx:end_idx]
            
            # Predizer
            if self.is_ensemble:
                pred = self._ensemble_predict(
                    n_predict, current_history, current_past_cov, iter_future_cov
                )
            else:
                pred = self._predict_single_model(
                    self.models[0], n_predict, current_history,
                    current_past_cov, iter_future_cov
                )
            
            # Armazenar predições
            all_predictions.append(pred)
            
            # Atualizar histórico
            current_history = current_history.append(pred)
            
            # Atualizar past_covariates se necessário
            if current_past_cov is not None and self.supports_past_covariates:
                current_past_cov = self._extend_covariates(
                    current_past_cov,
                    len(current_history)
                )
            
            # Manter apenas history_size mais recente
            if len(current_history) > self.history_size:
                current_history = current_history[-self.history_size:]
                if current_past_cov is not None:
                    current_past_cov = current_past_cov[-self.history_size:]
            
            steps_remaining -= n_predict
            step_offset += n_predict
        
        # Concatenar todas as predições
        if len(all_predictions) == 1:
            return all_predictions[0]
        
        result = all_predictions[0]
        for pred in all_predictions[1:]:
            result = result.append(pred)
        
        return result
    
    def _find_first_exceeding_step(
        self,
        hist: TimeSeries,
        target: float,
        past_cov: Optional[TimeSeries],
        future_cov: Optional[TimeSeries]
    ) -> Tuple[int, TimeSeries]:
        """
        Encontra o primeiro step onde o acumulado DAS PREDIÇÕES ultrapassa o target
        
        Returns
        -------
        tuple
            (n_steps, path) onde n_steps é o primeiro step que ultrapassa target
        
        Notes
        -----
        O acumulado é calculado APENAS sobre as predições (path), começando do zero.
        Não inclui o valor acumulado do histórico.
        """
        # Estimativa inicial para começar a busca
        avg_per_step = self._get_last_history_value(hist)
        if avg_per_step > 0:
            estimated_steps = max(1, int(np.ceil(target / avg_per_step)))
        else:
            estimated_steps = self.forecast_horizon
        
        # Busca binária para encontrar o primeiro step que ultrapassa
        left, right = 1, min(self.max_steps, max(estimated_steps * 3, 10))
        result_n_steps = None
        result_path = None
        
        while left <= right:
            mid = (left + right) // 2
            
            # Predizer
            pred_path = self._autoregressive_predict(
                hist, mid, past_cov, future_cov
            )
            
            # Acumulado APENAS das predições (começando do zero)
            accumulated_predictions = self._get_accumulated_value(pred_path)
            
            if accumulated_predictions >= target:
                # Ultrapassou - pode ser a resposta, mas vamos buscar menor
                result_n_steps = mid
                result_path = pred_path
                right = mid - 1
            else:
                # Ainda não ultrapassou - precisa de mais steps
                left = mid + 1
        
        if result_n_steps is None:
            # Não conseguiu atingir o target dentro de max_steps
            warnings.warn(
                f"Não foi possível atingir target {target:.2f} dentro de max_steps={self.max_steps}."
            )
            # Retornar max_steps
            result_path = self._autoregressive_predict(
                hist, self.max_steps, past_cov, future_cov
            )
            result_n_steps = self.max_steps
        
        return result_n_steps, result_path
    
    def _process_single_series_date_to_reach(
        self,
        idx: int,
        hist: TimeSeries,
        target: float,
        past_cov: Optional[TimeSeries],
        future_cov: Optional[TimeSeries]
    ) -> Tuple[pd.Timestamp, int, TimeSeries]:
        """Processa uma única série para date_to_reach_accumulated"""
        # Verificar se target é válido (positivo)
        if target <= 0:
            warnings.warn(f"Target deve ser positivo, recebido: {target}")
            return (
                hist.end_time(),
                0,
                TimeSeries.from_times_and_values(
                    times=[hist.end_time()],
                    values=[[0.0]]
                )
            )
        
        # Encontrar primeiro step que ultrapassa
        n_steps, path = self._find_first_exceeding_step(
            hist, target, past_cov, future_cov
        )
        
        predicted_date = hist.end_time() + n_steps * hist.freq
        return predicted_date, n_steps, path
    
    def _process_single_series_accumulated_at_step(
        self,
        idx: int,
        hist: TimeSeries,
        n_steps: Optional[int],
        ref_date: Optional[pd.Timestamp],
        past_cov: Optional[TimeSeries],
        future_cov: Optional[TimeSeries]
    ) -> Tuple[float, TimeSeries]:
        """Processa uma única série para predict_accumulated_at_step"""
        last_date = hist.end_time()
        freq = hist.freq
        
        # Determinar n_steps
        if n_steps is not None and ref_date is not None:
            # Ambos fornecidos: verificar consistência
            expected_date = last_date + n_steps * freq
            if expected_date != ref_date:
                raise ValueError(
                    f"Inconsistência: n_steps={n_steps} resulta em {expected_date}, "
                    f"mas reference_date={ref_date}"
                )
            final_n_steps = n_steps
            final_date = ref_date
        elif n_steps is not None:
            # Apenas n_steps fornecido
            final_n_steps = n_steps
            final_date = last_date + n_steps * freq
        elif ref_date is not None:
            # Apenas reference_date fornecido
            if ref_date <= last_date:
                warnings.warn(
                    f"reference_date ({ref_date}) está no passado ou presente do histórico ({last_date}). "
                    f"Retornando acumulado zero."
                )
                return (
                    0.0,
                    TimeSeries.from_times_and_values(
                        times=[ref_date],
                        values=[[0.0]]
                    )
                )
            final_n_steps = int((ref_date - last_date) / freq)
            final_date = ref_date
        else:
            raise ValueError("Deve fornecer n_steps ou reference_date (ou ambos)")
        
        # Validar n_steps
        if final_n_steps <= 0:
            warnings.warn(f"n_steps deve ser positivo, recebido: {final_n_steps}")
            return (
                0.0,
                TimeSeries.from_times_and_values(
                    times=[last_date],
                    values=[[0.0]]
                )
            )
        
        if final_n_steps > self.max_steps:
            warnings.warn(
                f"n_steps ({final_n_steps}) excede max_steps ({self.max_steps}). "
                f"Usando max_steps."
            )
            final_n_steps = self.max_steps
        
        # Predizer
        pred_path = self._autoregressive_predict(
            hist, final_n_steps, past_cov, future_cov
        )
        
        # Calcular acumulado APENAS das predições (começando do zero)
        accumulated_predictions = self._get_accumulated_value(pred_path)
        
        return accumulated_predictions, pred_path
    
    def date_to_reach_accumulated(
        self,
        history: Union[TimeSeries, List[TimeSeries]],
        target_value: Union[float, List[float]],
        past_covariates: Optional[Union[TimeSeries, List[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, List[TimeSeries]]] = None,
        n_jobs: int = 1
    ) -> Dict[str, Union[pd.Timestamp, List[pd.Timestamp], int, List[int], TimeSeries, List[TimeSeries]]]:
        """
        Encontra a PRIMEIRA data em que o valor acumulado DAS PREDIÇÕES ultrapassa target_value
        
        O acumulado é calculado APENAS sobre as predições futuras, começando do zero.
        Não inclui o valor acumulado do histórico.
        
        Parameters
        ----------
        history : TimeSeries ou List[TimeSeries]
            Histórico(s) de entrada
        target_value : float ou List[float]
            Valor(es) alvo acumulado(s) a serem atingidos pelas predições
        past_covariates : TimeSeries ou List[TimeSeries], optional
            Covariáveis passadas
        future_covariates : TimeSeries ou List[TimeSeries], optional
            Covariáveis futuras
        n_jobs : int, default=1
            Número de threads paralelas.
            - 1: sequencial
            - -1: usar todos os CPUs disponíveis
            - n > 1: usar n threads
        
        Returns
        -------
        Dict com:
            'date': pd.Timestamp ou List[pd.Timestamp]
                Primeira data que ultrapassa o target
            'n_steps': int ou List[int]
                Número de steps até ultrapassar o target
            'path': TimeSeries ou List[TimeSeries]
                Série(s) temporal(is) com as predições até ultrapassar o target
        
        Examples
        --------
        >>> # Exemplo: quero que o veículo acumule 10000 km a partir de agora
        >>> result = predictor.date_to_reach_accumulated(
        ...     history=ts,
        ...     target_value=10000.0  # 10000 km de predições futuras
        ... )
        >>> print(f"Acumulará 10000 km em {result['date']} ({result['n_steps']} dias)")
        >>> 
        >>> # O acumulado das predições (path) será >= 10000
        >>> accumulated_predictions = result['path'].values().sum()
        >>> print(f"Acumulado das predições: {accumulated_predictions:.2f} km")
        
        >>> # Batch com paralelização
        >>> result = predictor.date_to_reach_accumulated(
        ...     history=[ts1, ts2, ..., ts100],
        ...     target_value=[10000] * 100,
        ...     n_jobs=-1  # Usar todos os CPUs
        ... )
        """
        # Normalizar entradas
        history_list, target_list, past_cov_list, future_cov_list, is_single = self._normalize_inputs(
            history, target_value, past_covariates, future_covariates
        )
        
        n_series = len(history_list)
        
        # Decidir se vale a pena paralelizar
        use_parallel = n_jobs != 1 and n_series > 1
        
        if use_parallel:
            # Determinar número de workers
            if n_jobs == -1:
                n_workers = mp.cpu_count()
            else:
                n_workers = min(n_jobs, n_series)
            
            if self.verbose:
                print(f"🔄 Processando {n_series} séries com {n_workers} workers")
            
            # Preparar argumentos
            args_list = [
                (
                    idx,
                    hist,
                    target,
                    past_cov_list[idx] if past_cov_list else None,
                    future_cov_list[idx] if future_cov_list else None
                )
                for idx, (hist, target) in enumerate(zip(history_list, target_list))
            ]
            
            # Executar em paralelo usando threads
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                results = list(executor.map(
                    lambda args: self._process_single_series_date_to_reach(*args),
                    args_list
                ))
            
            dates, n_steps_list, paths = zip(*results)
            dates = list(dates)
            n_steps_list = list(n_steps_list)
            paths = list(paths)
            
        else:
            # Processamento sequencial
            dates = []
            n_steps_list = []
            paths = []
            
            for idx, (hist, target, past_cov, future_cov) in enumerate(
                zip(history_list, target_list,
                    past_cov_list or [None] * n_series,
                    future_cov_list or [None] * n_series)
            ):
                if self.verbose:
                    print(f"\n{'='*60}")
                    print(f"Série {idx+1}/{n_series}")
                    print(f"Target: {target:.2f}")
                    print(f"{'='*60}")
                
                date, n_steps, path = self._process_single_series_date_to_reach(
                    idx, hist, target, past_cov, future_cov
                )
                
                dates.append(date)
                n_steps_list.append(n_steps)
                paths.append(path)
                
                if self.verbose:
                    print(f"✓ Primeira data que ultrapassa: {date} ({n_steps} steps)")
        
        # Retornar formato apropriado
        if is_single:
            return {
                'date': dates[0],
                'n_steps': n_steps_list[0],
                'path': paths[0]
            }
        else:
            return {
                'date': dates,
                'n_steps': n_steps_list,
                'path': paths
            }
    
    def predict_accumulated_at_step(
        self,
        history: Union[TimeSeries, List[TimeSeries]],
        reference_date: Optional[Union[pd.Timestamp, List[pd.Timestamp]]] = None,
        n_steps: Optional[Union[int, List[int]]] = None,
        past_covariates: Optional[Union[TimeSeries, List[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, List[TimeSeries]]] = None,
        n_jobs: int = 1
    ) -> Dict[str, Union[float, List[float], TimeSeries, List[TimeSeries]]]:
        """
        Prediz o valor acumulado DAS PREDIÇÕES em uma data de referência ou após n_steps
        
        O acumulado é calculado APENAS sobre as predições futuras, começando do zero.
        Não inclui o valor acumulado do histórico.
        
        Parameters
        ----------
        history : TimeSeries ou List[TimeSeries]
            Histórico(s) de entrada
        reference_date : pd.Timestamp ou List[pd.Timestamp], optional
            Data(s) de referência. Pode ser fornecido junto com n_steps para validação.
        n_steps : int ou List[int], optional
            Número de steps para predizer. Pode ser fornecido junto com reference_date para validação.
        past_covariates : TimeSeries ou List[TimeSeries], optional
            Covariáveis passadas
        future_covariates : TimeSeries ou List[TimeSeries], optional
            Covariáveis futuras
        n_jobs : int, default=1
            Número de threads paralelas.
            - 1: sequencial
            - -1: usar todos os CPUs disponíveis
            - n > 1: usar n threads
        
        Returns
        -------
        Dict com:
            'accumulated_value': float ou List[float]
                Valor(es) acumulado(s) das predições
            'path': TimeSeries ou List[TimeSeries]
                Série(s) temporal(is) com as predições
        
        Raises
        ------
        ValueError
            Se nem reference_date nem n_steps forem fornecidos, ou se
            ambos forem fornecidos mas forem inconsistentes.
        
        Examples
        --------
        >>> # Exemplo: quanto o veículo vai acumular nos próximos 90 dias?
        >>> result = predictor.predict_accumulated_at_step(
        ...     history=ts,
        ...     n_steps=90
        ... )
        >>> print(f"Acumulará {result['accumulated_value']:.2f} km em 90 dias")
        >>> 
        >>> # O valor retornado é APENAS das predições (path)
        >>> assert result['accumulated_value'] == result['path'].values().sum()
        
        >>> # Usando reference_date
        >>> result = predictor.predict_accumulated_at_step(
        ...     history=ts,
        ...     reference_date=pd.Timestamp('2024-12-31')
        ... )
        
        >>> # Usando ambos (validação de consistência)
        >>> result = predictor.predict_accumulated_at_step(
        ...     history=ts,
        ...     reference_date=pd.Timestamp('2024-12-31'),
        ...     n_steps=90  # Deve ser consistente com reference_date
        ... )
        
        >>> # Batch com paralelização
        >>> result = predictor.predict_accumulated_at_step(
        ...     history=[ts1, ts2, ..., ts100],
        ...     n_steps=90,
        ...     n_jobs=-1  # Usar todos os CPUs
        ... )
        """
        # Validar entrada
        if reference_date is None and n_steps is None:
            raise ValueError("Deve fornecer reference_date ou n_steps")
        
        # Normalizar history
        if isinstance(history, TimeSeries):
            history_list = [history]
            is_single = True
        else:
            history_list = history
            is_single = False
        
        n_series = len(history_list)
        
        # Normalizar reference_date
        if reference_date is None:
            date_list = [None] * n_series
        elif isinstance(reference_date, pd.Timestamp):
            if is_single:
                date_list = [reference_date]
            else:
                date_list = [reference_date] * n_series
        else:
            date_list = list(reference_date)
            if len(date_list) != n_series:
                raise ValueError(
                    f"reference_date deve ter mesmo comprimento que history "
                    f"({len(date_list)} != {n_series})"
                )
        
        # Normalizar n_steps
        if n_steps is None:
            steps_list = [None] * n_series
        elif isinstance(n_steps, int):
            if is_single:
                steps_list = [n_steps]
            else:
                steps_list = [n_steps] * n_series
        else:
            steps_list = list(n_steps)
            if len(steps_list) != n_series:
                raise ValueError(
                    f"n_steps deve ter mesmo comprimento que history "
                    f"({len(steps_list)} != {n_series})"
                )
        
        # Normalizar covariates
        _, _, past_cov_list, future_cov_list, _ = self._normalize_inputs(
            history, [0.0] * n_series, past_covariates, future_covariates
        )
        
        # Decidir se vale a pena paralelizar
        use_parallel = n_jobs != 1 and n_series > 1
        
        if use_parallel:
            # Determinar número de workers
            if n_jobs == -1:
                n_workers = mp.cpu_count()
            else:
                n_workers = min(n_jobs, n_series)
            
            if self.verbose:
                print(f"🔄 Processando {n_series} séries com {n_workers} workers")
            
            # Preparar argumentos
            args_list = [
                (
                    idx,
                    hist,
                    steps_list[idx],
                    date_list[idx],
                    past_cov_list[idx] if past_cov_list else None,
                    future_cov_list[idx] if future_cov_list else None
                )
                for idx, hist in enumerate(history_list)
            ]
            
            # Executar em paralelo usando threads
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                results = list(executor.map(
                    lambda args: self._process_single_series_accumulated_at_step(*args),
                    args_list
                ))
            
            accumulated_values, paths = zip(*results)
            accumulated_values = list(accumulated_values)
            paths = list(paths)
            
        else:
            # Processamento sequencial
            accumulated_values = []
            paths = []
            
            for idx, (hist, n_step, ref_date, past_cov, future_cov) in enumerate(
                zip(history_list, steps_list, date_list,
                    past_cov_list or [None] * n_series,
                    future_cov_list or [None] * n_series)
            ):
                if self.verbose:
                    print(f"\n{'='*60}")
                    print(f"Série {idx+1}/{n_series}")
                    if ref_date:
                        print(f"Data referência: {ref_date}")
                    if n_step:
                        print(f"N steps: {n_step}")
                    print(f"{'='*60}")
                
                value, path = self._process_single_series_accumulated_at_step(
                    idx, hist, n_step, ref_date, past_cov, future_cov
                )
                
                accumulated_values.append(value)
                paths.append(path)
                
                if self.verbose:
                    print(f"✓ Valor acumulado: {value:.2f}")
        
        # Retornar formato apropriado
        if is_single:
            return {
                'accumulated_value': accumulated_values[0],
                'path': paths[0]
            }
        else:
            return {
                'accumulated_value': accumulated_values,
                'path': paths
            }
    
    def __repr__(self) -> str:
        if self.is_ensemble:
            return (
                f"Predictor(\n"
                f"  ensemble={len(self.models)} modelos,\n"
                f"  weights={self.model_weights},\n"
                f"  history_size={self.history_size},\n"
                f"  forecast_horizon={self.forecast_horizon},\n"
                f"  max_steps={self.max_steps}\n"
                f")"
            )
        else:
            return (
                f"Predictor(\n"
                f"  model={self.models[0].__class__.__name__},\n"
                f"  history_size={self.history_size},\n"
                f"  forecast_horizon={self.forecast_horizon},\n"
                f"  max_steps={self.max_steps}\n"
                f")"
            )
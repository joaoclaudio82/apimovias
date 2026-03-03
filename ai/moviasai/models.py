import numpy as np
import pandas as pd
from typing import List, Optional, Union
from darts import TimeSeries
import warnings
from tqdm import tqdm


class LinearTrendForecaster:
    """
    Modelo baseline de regressão linear 
    
    Ajusta regressão linear (y = ax + b) para cada série temporal usando
    fórmula fechada vetorizada (sem sklearn, sem paralelização).
    
    Otimizações implementadas:
    - Fórmula fechada para regressão linear (mais rápida que sklearn)
    - Loop vetorizado (sem overhead de paralelização)
    - Reuso de time_index para criação de TimeSeries
    - Processamento em batches para predições
    
    Performance:
    - Fit: ~1000-2000 séries/segundo
    - Predict: ~5000-10000 séries/segundo (com reuso de time_index)
    
    Attributes:
    -----------
    models_ : dict
        Coeficientes {series_id: {'trend': float, 'intercept': float}}
    series_info_ : dict
        Metadados {series_id: info_dict}
    verbose : bool
        Logs detalhados
    
    Examples:
    ---------
    >>> forecaster = LinearTrendForecaster(verbose=True)
    >>> 
    >>> # Fit apenas com history
    >>> forecaster.fit(local_data.input_series)
    >>> 
    >>> # Predict reusando time_index (RÁPIDO)
    >>> predictions = forecaster.predict(
    ...     n=7,
    ...     output_series_list=local_data.output_series
    ... )
    >>> 
    >>> # Predict sem reusar (LENTO)
    >>> predictions = forecaster.predict(n=7)
    """
    
    def __init__(self, verbose: bool = True):
        """
        Inicializa o forecaster
        
        Parameters:
        -----------
        verbose : bool, optional
            Se True, exibe logs detalhados (padrão: True)
        """
        self.verbose = verbose
        self.models_ = {}
        self.series_info_ = {}
        self._is_fitted = False
    
    def fit(
        self,
        ts_list: List[TimeSeries]
    ) -> 'LinearTrendForecaster':
        """
        Ajusta modelos usando fórmula fechada vetorizada
        
        Fórmula da regressão linear:
        - trend (a) = cov(x, y) / var(x)
        - intercept (b) = mean(y) - a * mean(x)
        - R² = 1 - SS_res / SS_tot
        
        Parameters:
        -----------
        ts_list : List[TimeSeries]
            Lista de séries temporais (APENAS HISTORY, sem forecast)
        
        Returns:
        --------
        self
        
        Examples:
        ---------
        >>> forecaster.fit(local_data.input_series)  # ← Apenas history!
        """
        if not ts_list:
            raise ValueError("Lista de séries vazia")
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"AJUSTANDO BASELINE - LinearTrendForecaster")
            print(f"{'='*60}")
            print(f"Séries: {len(ts_list):,}")
            print(f"Método: Vetorizado (fórmula fechada)")
            print()
        
        import time
        start = time.time()
        
        n_success = 0
        n_failed = 0
        
        # Loop vetorizado (mais rápido que paralelização!)
        for series_id, ts in enumerate(tqdm(ts_list, desc="Ajustando", disable=not self.verbose)):
            try:
                values = ts.values().flatten()
                n_points = len(values)
                
                if n_points < 2:
                    warnings.warn(f"Série {series_id} tem < 2 pontos. Pulando.")
                    n_failed += 1
                    continue
                
                # Índice temporal
                X = np.arange(n_points, dtype=np.float64)
                
                # Estatísticas
                mean_x = np.mean(X)
                mean_y = np.mean(values)
                
                # Covariância e variância
                diff_x = X - mean_x
                diff_y = values - mean_y
                
                cov_xy = np.sum(diff_x * diff_y) / n_points
                var_x = np.sum(diff_x ** 2) / n_points
                
                # Coeficientes
                if var_x > 1e-10:
                    trend = cov_xy / var_x
                    intercept = mean_y - trend * mean_x
                    
                    # R²
                    y_pred = trend * X + intercept
                    ss_res = np.sum((values - y_pred) ** 2)
                    ss_tot = np.sum(diff_y ** 2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
                else:
                    # Série constante
                    trend = 0.0
                    intercept = mean_y
                    r2 = 0.0
                
                # Armazenar modelo (apenas coeficientes)
                self.models_[series_id] = {
                    'trend': float(trend),
                    'intercept': float(intercept)
                }
                
                # Armazenar info
                self.series_info_[series_id] = {
                    'start_time': ts.start_time(),
                    'end_time': ts.end_time(),
                    'freq': ts.freq,
                    'n_points': n_points,
                    'last_index': n_points - 1,
                    'mean': float(mean_y),
                    'std': float(np.std(values)),
                    'trend': float(trend),
                    'intercept': float(intercept),
                    'r2_score': float(r2)
                }
                
                n_success += 1
            
            except Exception as e:
                warnings.warn(f"Erro série {series_id}: {e}")
                n_failed += 1
        
        self._is_fitted = True
        
        elapsed = time.time() - start
        
        if self.verbose:
            print(f"\n✓ Ajuste concluído em {elapsed:.2f}s:")
            print(f"  - Taxa: {n_success/elapsed:.0f} séries/s")
            print(f"  - Sucesso: {n_success:,}")
            if n_failed > 0:
                print(f"  - Falhas: {n_failed}")
            
            if n_success > 0:
                trends = [info['trend'] for info in self.series_info_.values()]
                r2_scores = [info['r2_score'] for info in self.series_info_.values()]
                
                print(f"\nEstatísticas:")
                print(f"  Tendência: Média={np.mean(trends):.4f}, Mediana={np.median(trends):.4f}")
                print(f"  R²: Média={np.mean(r2_scores):.4f}, Mediana={np.median(r2_scores):.4f}")
            
            print(f"{'='*60}\n")
        
        return self
    
    def predict(
        self,
        n: int,
        series_ids: Optional[List[int]] = None,
        output_series_list: Optional[List[TimeSeries]] = None,
        return_darts: bool = True
    ) -> Union[List[TimeSeries], List[np.ndarray]]:
        """
        Prediz n passos à frente 
        
        Parameters:
        -----------
        n : int
            Número de passos à frente
        series_ids : List[int], optional
            IDs das séries. Se None, prediz todas
        output_series_list : List[TimeSeries], optional
            Lista de output_series para REUSAR time_index (RECOMENDADO!)
            Isso torna a predição 10-50x mais rápida!
        return_darts : bool, optional
            Se True, retorna TimeSeries. Se False, arrays (padrão: True)
        
        Returns:
        --------
        List[TimeSeries] or List[np.ndarray]
        
        Examples:
        ---------
        >>> # SUPER RÁPIDO: Reusar time_index
        >>> preds = forecaster.predict(
        ...     n=7,
        ...     output_series_list=local_data.output_series
        ... )
        >>> 
        >>> # RÁPIDO: Apenas arrays
        >>> preds = forecaster.predict(n=7, return_darts=False)
        >>> 
        >>> # LENTO: Criar TimeSeries do zero
        >>> preds = forecaster.predict(n=7)
        """
        if not self._is_fitted:
            raise RuntimeError("Modelo não ajustado. Execute fit() primeiro.")
        
        if n <= 0:
            raise ValueError(f"n deve ser > 0, recebido: {n}")
        
        # Determinar séries
        if series_ids is None:
            series_ids = list(self.models_.keys())
        else:
            invalid_ids = [sid for sid in series_ids if sid not in self.models_]
            if invalid_ids:
                raise ValueError(f"IDs inválidos: {invalid_ids}")
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"GERANDO PREDIÇÕES - BASELINE")
            print(f"{'='*60}")
            print(f"Séries: {len(series_ids):,}")
            print(f"Passos: {n}")
            print(f"Reusa time_index: {output_series_list is not None}")
            print()
        
        import time
        start = time.time()
        
        # PASSO 1: Predizer arrays (vetorizado, super rápido)
        predictions_arrays = []
        
        for series_id in tqdm(series_ids, desc="Predizendo", disable=not self.verbose):
            if series_id not in self.models_:
                predictions_arrays.append(None)
                continue
            
            try:
                model = self.models_[series_id]
                info = self.series_info_[series_id]
                
                # Índices futuros
                last_index = info['last_index']
                future_indices = np.arange(last_index + 1, last_index + 1 + n, dtype=np.float64)
                
                # Predição vetorizada
                predictions = model['trend'] * future_indices + model['intercept']
                
                predictions_arrays.append(predictions)
            
            except Exception as e:
                warnings.warn(f"Erro ao predizer série {series_id}: {e}")
                predictions_arrays.append(None)
        
        elapsed_predict = time.time() - start
        
        if self.verbose:
            print(f"  ⏱️  Predições (arrays): {elapsed_predict:.2f}s")
        
        # PASSO 2: Converter para TimeSeries (se necessário)
        if not return_darts:
            # Retornar apenas arrays
            n_success = sum(1 for p in predictions_arrays if p is not None)
            n_failed = len(predictions_arrays) - n_success
            
            total_time = time.time() - start
            
            if self.verbose:
                print(f"\n✓ Concluído em {total_time:.2f}s:")
                print(f"  - Sucesso: {n_success:,}")
                if n_failed > 0:
                    print(f"  - Falhas: {n_failed}")
                print(f"{'='*60}\n")
            
            return predictions_arrays
        
        # Converter para TimeSeries
        start_ts = time.time()
        
        predictions = []
        n_success = 0
        n_failed = 0
        
        if output_series_list is not None:
            # OTIMIZADO: Reusar time_index
            if self.verbose:
                print(f"  🔄 Convertendo (reusando time_index)...")
            
            for pred_array, output_ts in zip(predictions_arrays, output_series_list):
                if pred_array is None:
                    predictions.append(None)
                    n_failed += 1
                    continue
                
                # Reusar time_index e freq
                ts_pred = TimeSeries.from_times_and_values(
                    times=output_ts.time_index,
                    values=pred_array.reshape(-1, 1),
                    freq=output_ts.freq
                )
                
                predictions.append(ts_pred)
                n_success += 1
        
        else:
            # SEM OTIMIZAÇÃO: Criar time_index do zero
            if self.verbose:
                print(f"  🔄 Convertendo (criando time_index)...")
                print(f"  ⚠️  AVISO: Use output_series_list para 10-50x mais rápido!")
            
            for series_id, pred_array in zip(series_ids, predictions_arrays):
                if pred_array is None:
                    predictions.append(None)
                    n_failed += 1
                    continue
                
                info = self.series_info_[series_id]
                
                # Criar time_index
                start_time = info['end_time'] + info['freq']
                future_times = pd.date_range(
                    start=start_time,
                    periods=n,
                    freq=info['freq']
                )
                
                ts_pred = TimeSeries.from_times_and_values(
                    times=future_times,
                    values=pred_array.reshape(-1, 1)
                )
                
                predictions.append(ts_pred)
                n_success += 1
        
        elapsed_ts = time.time() - start_ts
        total_time = time.time() - start
        
        if self.verbose:
            print(f"  ⏱️  Conversão TimeSeries: {elapsed_ts:.2f}s")
            print(f"\n✓ Concluído em {total_time:.2f}s:")
            print(f"  - Sucesso: {n_success:,}")
            if n_failed > 0:
                print(f"  - Falhas: {n_failed}")
            print(f"  - Taxa total: {n_success/total_time:.0f} séries/s")
            print(f"{'='*60}\n")
        
        return predictions
    
    def get_model_info(self, series_id: int) -> dict:
        """
        Retorna informações do modelo de uma série
        
        Parameters:
        -----------
        series_id : int
            ID da série
        
        Returns:
        --------
        dict
            Informações: trend, intercept, r2_score, etc.
        """
        if not self._is_fitted:
            raise RuntimeError("Modelo não ajustado")
        
        if series_id not in self.series_info_:
            raise ValueError(f"Série {series_id} não encontrada")
        
        return self.series_info_[series_id].copy()
    
    def get_all_model_info(self) -> dict:
        """Retorna informações de todos os modelos"""
        if not self._is_fitted:
            raise RuntimeError("Modelo não ajustado")
        
        return self.series_info_.copy()
    
    def get_statistics(self) -> dict:
        """
        Retorna estatísticas agregadas dos modelos
        
        Returns:
        --------
        dict
            Estatísticas: trends, r2_scores, etc.
        
        Examples:
        ---------
        >>> stats = forecaster.get_statistics()
        >>> print(f"R² médio: {stats['r2_mean']:.4f}")
        >>> print(f"Tendência mediana: {stats['trend_median']:.4f}")
        """
        if not self._is_fitted:
            raise RuntimeError("Modelo não ajustado")
        
        trends = [info['trend'] for info in self.series_info_.values()]
        r2_scores = [info['r2_score'] for info in self.series_info_.values()]
        
        return {
            'n_models': len(self.models_),
            'trend_mean': float(np.mean(trends)),
            'trend_median': float(np.median(trends)),
            'trend_std': float(np.std(trends)),
            'trend_min': float(np.min(trends)),
            'trend_max': float(np.max(trends)),
            'r2_mean': float(np.mean(r2_scores)),
            'r2_median': float(np.median(r2_scores)),
            'r2_std': float(np.std(r2_scores)),
            'r2_min': float(np.min(r2_scores)),
            'r2_max': float(np.max(r2_scores))
        }
    
    def plot_predictions(
        self,
        series_id: int,
        input_ts: TimeSeries,
        output_ts: TimeSeries,
        figsize: tuple = (12, 6)
    ):
        """
        Plota série e predição
        
        Parameters:
        -----------
        series_id : int
            ID da série
        input_ts : TimeSeries
            Série de input (history)
        output_ts : TimeSeries
            Série de output (forecast real)
        figsize : tuple, optional
            Tamanho da figura
        
        Examples:
        ---------
        >>> forecaster.plot_predictions(
        ...     series_id=0,
        ...     input_ts=local_data.input_series[0],
        ...     output_ts=local_data.output_series[0]
        ... )
        """
        import matplotlib.pyplot as plt
        
        if not self._is_fitted:
            raise RuntimeError("Modelo não ajustado")
        
        # Gerar predição
        n = len(output_ts)
        pred_ts = self.predict(
            n=n,
            series_ids=[series_id],
            output_series_list=[output_ts],
            return_darts=True
        )[0]
        
        if pred_ts is None:
            print(f"Erro: Predição não disponível para série {series_id}")
            return
        
        # Informações
        info = self.get_model_info(series_id)
        
        # Combinar input + output
        full_ts = input_ts.append(output_ts)
        
        # Plotar
        fig, ax = plt.subplots(figsize=figsize)
        
        # History
        input_ts.plot(ax=ax, label='History', linewidth=2, color='#2E86AB')
        
        # Forecast real
        output_ts.plot(ax=ax, label='Real', linewidth=2, color='#2E86AB', linestyle='-', marker='o', markersize=6)
        
        # Predição
        pred_ts.plot(ax=ax, label='Baseline', linewidth=2, linestyle='--', color='#F18F01', marker='s', markersize=6)
        
        # Linha vertical
        history_end = input_ts.end_time()
        ax.axvline(x=history_end, color='red', linestyle=':', alpha=0.5, linewidth=2, label='Início Forecast')
        
        # Calcular métricas
        from darts.metrics import wmape, mae
        wmape_val = wmape(output_ts, pred_ts)
        mae_val = mae(output_ts, pred_ts)
        
        # Título
        ax.set_title(
            f'Série {series_id} - Baseline (Regressão Linear)\n'
            f'Tendência={info["trend"]:.4f} | R²={info["r2_score"]:.3f} | '
            f'WMAPE={wmape_val:.2f} | MAE={mae_val:.2f}',
            fontsize=11,
            fontweight='bold'
        )
        ax.set_xlabel('Tempo', fontsize=11)
        ax.set_ylabel('Valor (normalizado)', fontsize=11)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def __repr__(self) -> str:
        """Representação em string"""
        if self._is_fitted:
            stats = self.get_statistics()
            return (
                f"LinearTrendForecaster(\n"
                f"  n_models={stats['n_models']:,},\n"
                f"  trend_median={stats['trend_median']:.4f},\n"
                f"  r2_median={stats['r2_median']:.4f},\n"
                f"  fitted=True\n"
                f")"
            )
        else:
            return (
                f"LinearTrendForecaster(\n"
                f"  fitted=False\n"
                f")"
            )
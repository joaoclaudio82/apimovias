import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Any
import warnings
import re

from darts.metrics import wmape, smape, r2_score, mae, mse
from darts import TimeSeries

from dataset_loader import DatasetLoader
from vehicle_profile import VehicleProfile
from train import MetricStats
from models import LinearTrendForecaster

import pickle
import hashlib
import torch

from darts.models import (
    DLinearModel, NLinearModel, TSMixerModel, TCNModel,
    NBEATSModel, NHiTSModel, LightGBMModel, XGBModel,
    LinearRegressionModel
)
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel

class BaselineEvaluator:
    """
    Avaliador do modelo baseline (LinearTrendForecaster) em datasets locais
    
    Avalia o baseline em todos os datasets locais (não merged) de um target,
    calculando métricas por classe e agregadas, com análise de performance
    diária e acumulada.
    """
    
    def __init__(
        self,
        local_datasets_dir: str,
        vp: VehicleProfile,
        baseline_model: LinearTrendForecaster,
        history_sizes: Optional[List[int]] = None,
        cache_dir: str = '.cache/windows',
        n_jobs: int = -1,
        verbose: bool = True
    ):
        """
        Inicializa o avaliador de baseline
        
        Parameters:
        -----------
        local_datasets_dir : str
            Diretório raiz com datasets locais
            Ex: '../datasets/local'
        vp : VehicleProfile
            Perfil de veículos
        baseline_model : LinearTrendForecaster
            Modelo baseline a avaliar
        history_sizes : List[int], optional
            Tamanhos de histórico a avaliar. Se None, descobre automaticamente
        cache_dir : str, optional
            Diretório de cache (padrão: '.cache/windows')
        n_jobs : int, optional
            Jobs paralelos (padrão: -1)
        verbose : bool, optional
            Logs detalhados (padrão: True)
        
        Examples:
        ---------
        >>> from vehicle_profile import VehicleProfile
        >>> 
        >>> vp = VehicleProfile.load('../profiles/km')
        >>> baseline = LinearTrendForecaster(n_jobs=-1)
        >>> 
        >>> evaluator = BaselineEvaluator(
        ...     local_datasets_dir='../datasets/local',
        ...     vp=vp,
        ...     baseline_model=baseline,
        ...     verbose=True
        ... )
        >>> 
        >>> evaluator.evaluate_all()
        >>> evaluator.print_summary()
        >>> evaluator.compare_classes()
        """
        self.local_datasets_dir = Path(local_datasets_dir)
        self.vp = vp
        self.baseline_model = baseline_model
        self.cache_dir = cache_dir
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        self.target = vp.target
        
        if verbose:
            print("=" * 80)
            print("📊 INICIALIZANDO AVALIAÇÃO DE BASELINE")
            print("=" * 80)
            print(f"Local datasets dir: {self.local_datasets_dir}")
            print(f"Target: {self.target}")
            print(f"Baseline model: {baseline_model.__class__.__name__}")
            print()
        
        # Descobrir datasets locais
        self.local_datasets = self._discover_local_datasets(history_sizes)
        
        if not self.local_datasets:
            raise ValueError(f"Nenhum dataset local encontrado para target '{self.target}'")
        
        if verbose:
            print(f"✓ {len(self.local_datasets)} dataset(s) local(is) encontrado(s):")
            for ds_info in self.local_datasets:
                print(f"  • {ds_info['dataset_name']} (classe {ds_info['class_id']}, hist={ds_info['history_size']})")
            print()
        
        # Resultados
        self.predictions = {}  # {(dataset_name, class_id): {history_size: List[TimeSeries]}}
        self.metrics = {}      # {(dataset_name, class_id): {history_size: pd.DataFrame}}
    
    def _discover_local_datasets(
        self,
        history_sizes: Optional[List[int]]
    ) -> List[Dict[str, Any]]:
        """
        Descobre datasets locais (não merged) do target
        
        Returns:
        --------
        List[dict]
            Lista com: dataset_name, class_id, history_size, split_name, path
        """
        datasets = []
        
        # Padrão: dataset_{target}_*
        pattern = f"dataset_{self.target}_*"
        
        for dataset_path in self.local_datasets_dir.glob(pattern):
            if not dataset_path.is_dir():
                continue
            
            dataset_name = dataset_path.name
            match_hist = re.search(r'hist(\d+)', dataset_name)
            if not match_hist:
                continue    
            
            hist_size = int(match_hist.group(1))

            # Filtrar por history_sizes se especificado
            if history_sizes is not None and hist_size not in history_sizes:
                continue

            for class_split in dataset_path.glob('class*.csv'):
                # Extrair class_id
                match_class = re.search(r'class(\d+)', class_split.name)
                class_id = int(match_class.group(1)) if match_class else None
                                    
                split_name = class_split.stem 
                
                datasets.append({
                    'dataset_name': dataset_name,
                    'class_id': class_id,
                    'history_size': hist_size,
                    'split_name': split_name,
                    'path': dataset_path
                })
        
        # Ordenar por class_id e history_size
        datasets.sort(key=lambda x: (x['class_id'] or -1, x['history_size']))
        
        return datasets
    
    def evaluate_all(self):
        """Avalia baseline em todos os datasets locais"""
        if self.verbose:
            print("=" * 80)
            print("🔮 AVALIANDO BASELINE")
            print("=" * 80)
            print()
        
        for ds_info in self.local_datasets:
            dataset_name = ds_info['dataset_name']
            class_id = ds_info['class_id']
            history_size = ds_info['history_size']
            split_name = ds_info['split_name']
            
            if self.verbose:
                print(f"\n{'-'*80}")
                print(f"Dataset: {dataset_name}")
                print(f"Classe: {class_id} | History: {history_size}")
                print(f"{'-'*80}")
            
            try:
                # Carregar dataset local
                loader = DatasetLoader(
                    dataset_dir=ds_info['path'],
                    vp=self.vp,
                    verbose=False
                )
                
                local_windows = loader.load(
                    split=split_name,
                    return_windows=True, 
                    use_cache=True,
                    cache_dir=self.cache_dir
                )
                
                if self.verbose:
                    n_series_total = sum(len(w) for w in local_windows)
                    print(f"  ✓ {len(local_windows)} janela(s) carregada(s)")
                    print(f"  ✓ {n_series_total} séries totais")
                    print(f"  ✓ Forecast horizon: {loader.forecast_horizon} dias")
                
                # Processar cada janela
                all_predictions = []
                all_metrics = []
                
                for window_idx, window in enumerate(local_windows):
                    if self.verbose:
                        print(f"  Janela {window_idx + 1}/{len(local_windows)}: {len(window)} séries")
                    
                    # Ajustar baseline 
                    self.baseline_model.fit(window.input_series)
                    
                    # Predizer (REUSANDO time_index)
                    predictions = self.baseline_model.predict(
                        n=loader.forecast_horizon,
                        output_series_list=window.output_series,
                        return_darts=True
                    )
                    
                    all_predictions.append(predictions)
                    
                    # Calcular métricas para esta janela
                    metrics_data = {
                        'veiculo_id': [],
                        'window': [],
                        'class': [],
                        'wmape': [],
                        'smape': [],
                        'r2': [],
                        'mae': [],
                        'mse': [],
                        'real_acc': [],
                        'pred_acc': [],
                        'real_daily_avg': [],
                        'pred_daily_avg': []
                    }
                    
                    # Iterar sobre séries da janela
                    for i, (real_ts, pred_ts) in enumerate(zip(window.output_series, predictions)):
                        if pred_ts is None:
                            continue
                        
                        vehicle_id = window.ids[i]
                        
                        # Métricas normalizadas
                        try:
                            wmape_val = float(wmape(real_ts, pred_ts))
                            smape_val = float(smape(real_ts, pred_ts))
                            r2_val = float(r2_score(real_ts, pred_ts))
                            mae_val = float(mae(real_ts, pred_ts))
                            mse_val = float(mse(real_ts, pred_ts))
                        except Exception as e:
                            warnings.warn(f"Erro ao calcular métricas série {i}: {e}")
                            wmape_val = np.nan
                            smape_val = np.nan
                            r2_val = np.nan
                            mae_val = np.nan
                            mse_val = np.nan
                        
                        # Desnormalizar
                        try:
                            history_end_date = window.input_series[i].end_time()
                            
                            real_values = self.vp.inverse_transform(
                                vehicle_id=vehicle_id,
                                scaled_values=real_ts.values().flatten(),
                                date=history_end_date
                            )
                            
                            pred_values = self.vp.inverse_transform(
                                vehicle_id=vehicle_id,
                                scaled_values=pred_ts.values().flatten(),
                                date=history_end_date
                            )
                            
                            real_acc = float(real_values.sum())
                            pred_acc = float(pred_values.sum())
                            real_daily = float(real_values.mean())
                            pred_daily = float(pred_values.mean())
                        
                        except Exception as e:
                            warnings.warn(f"Erro ao desnormalizar veículo {vehicle_id}: {e}")
                            real_acc = np.nan
                            pred_acc = np.nan
                            real_daily = np.nan
                            pred_daily = np.nan
                        
                        # Adicionar
                        metrics_data['veiculo_id'].append(vehicle_id)
                        metrics_data['window'].append(window_idx)
                        metrics_data['class'].append(class_id)
                        metrics_data['wmape'].append(wmape_val)
                        metrics_data['smape'].append(smape_val)
                        metrics_data['r2'].append(r2_val)
                        metrics_data['mae'].append(mae_val)
                        metrics_data['mse'].append(mse_val)
                        metrics_data['real_acc'].append(real_acc)
                        metrics_data['pred_acc'].append(pred_acc)
                        metrics_data['real_daily_avg'].append(real_daily)
                        metrics_data['pred_daily_avg'].append(pred_daily)
                    
                    df_metrics = pd.DataFrame(metrics_data)
                    all_metrics.append(df_metrics)
                    
                    if self.verbose:
                        print(f"    ✓ {len(df_metrics)} séries avaliadas")
                
                # Consolidar métricas de todas as janelas
                df_metrics_consolidated = pd.concat(all_metrics, ignore_index=True)
                
                data_key = (dataset_name, class_id)
                # Armazenar
                if data_key not in self.predictions:
                    self.predictions[data_key] = {}
                    self.metrics[data_key] = {}
                
                self.predictions[data_key][history_size] = all_predictions
                self.metrics[data_key][history_size] = df_metrics_consolidated
                
                if self.verbose:
                    print(f"  ✓ Total: {len(df_metrics_consolidated)} séries avaliadas")
            
            except Exception as e:
                print(f"  ❌ Erro: {e}")
        
        if self.verbose:
            print("\n" + "=" * 80)
            print()
            self.print_summary()

        
    def print_summary(self):
        """Imprime resumo completo - POR CLASSE E HISTORY_SIZE"""
        print("=" * 80)
        print("📊 RESUMO - BASELINE (LinearTrendForecaster)")
        print("=" * 80)
        print()
        
        # Agrupar por classe E history_size
        metrics_by_class_hist = {}
        
        for _, hist_metrics in self.metrics.items():
            for history_size, df_metrics in hist_metrics.items():
                class_id = df_metrics['class'].iloc[0]
                
                key = (class_id, history_size)
                
                if key not in metrics_by_class_hist:
                    metrics_by_class_hist[key] = []
                
                metrics_by_class_hist[key].append(df_metrics)
        
        # Imprimir por classe e history
        for (class_id, history_size) in sorted(metrics_by_class_hist.keys()):
            print(f"Classe {class_id} | History {history_size}:")
            print("-" * 80)
            
            df_list = metrics_by_class_hist[(class_id, history_size)]
            df_combined = pd.concat(df_list, ignore_index=True)
            
            # Agregar métricas
            all_wmape = df_combined['wmape'].values
            all_wmape = all_wmape[np.isfinite(all_wmape)]
            
            # MAPE acumulado
            valid_acc = ~(pd.isna(df_combined['real_acc']) | pd.isna(df_combined['pred_acc'])) & (df_combined['real_acc'] > 0)
            mape_acc = 100 * np.abs(
                df_combined.loc[valid_acc, 'real_acc'] - df_combined.loc[valid_acc, 'pred_acc']
            ) / df_combined.loc[valid_acc, 'real_acc']
            mape_acc = mape_acc[np.isfinite(mape_acc)]
            
            # MAE acumulado
            mae_acc = np.abs(
                df_combined.loc[valid_acc, 'real_acc'] - df_combined.loc[valid_acc, 'pred_acc']
            )
            mae_acc = mae_acc[np.isfinite(mae_acc)]
            
            # MAPE diário
            valid_daily = ~(pd.isna(df_combined['real_daily_avg']) | pd.isna(df_combined['pred_daily_avg'])) & (df_combined['real_daily_avg'] > 0)
            mape_daily = 100 * np.abs(
                df_combined.loc[valid_daily, 'real_daily_avg'] - df_combined.loc[valid_daily, 'pred_daily_avg']
            ) / df_combined.loc[valid_daily, 'real_daily_avg']
            mape_daily = mape_daily[np.isfinite(mape_daily)]
            
            # MAE diário
            mae_daily = np.abs(
                df_combined.loc[valid_daily, 'real_daily_avg'] - df_combined.loc[valid_daily, 'pred_daily_avg']
            )
            mae_daily = mae_daily[np.isfinite(mae_daily)]
            
            print(f"  Séries: {len(df_combined)}")
            print()
            
            if len(all_wmape) > 0:
                wmape_stats = MetricStats.from_array(all_wmape)
                print(f"  WMAPE (normalizado):")
                print(f"    Mediana: {wmape_stats.median:.4f} | P90: {wmape_stats.p90:.4f} | P95: {wmape_stats.p95:.4f}")
            
            if len(mape_acc) > 0:
                mape_acc_stats = MetricStats.from_array(mape_acc)
                print(f"  MAPE Acumulado:")
                print(f"    Mediana: {mape_acc_stats.median:.2f}% | P90: {mape_acc_stats.p90:.2f}% | P95: {mape_acc_stats.p95:.2f}%")
            
            if len(mae_acc) > 0:
                print(f"  MAE Acumulado:")
                print(f"    Mediana: {np.median(mae_acc):.2f} | P90: {np.percentile(mae_acc, 90):.2f}")
            
            if len(mape_daily) > 0:
                mape_daily_stats = MetricStats.from_array(mape_daily)
                print(f"  MAPE Diário:")
                print(f"    Mediana: {mape_daily_stats.median:.2f}% | P90: {mape_daily_stats.p90:.2f}% | P95: {mape_daily_stats.p95:.2f}%")
            
            if len(mae_daily) > 0:
                print(f"  MAE Diário:")
                print(f"    Mediana: {np.median(mae_daily):.2f} | P90: {np.percentile(mae_daily, 90):.2f}")
            
            print()
        
        # AGREGADO POR CLASSE (todos os history_sizes)
        print("\n" + "=" * 80)
        print("AGREGADO POR CLASSE (todos os history_sizes)")
        print("=" * 80)
        
        metrics_by_class_only = {}
        
        for (class_id, history_size), df_list in metrics_by_class_hist.items():
            if class_id not in metrics_by_class_only:
                metrics_by_class_only[class_id] = []
            metrics_by_class_only[class_id].extend(df_list)
        
        for class_id in sorted(metrics_by_class_only.keys()):
            print(f"\nClasse {class_id}:")
            print("-" * 80)
            
            df_combined = pd.concat(metrics_by_class_only[class_id], ignore_index=True)
            
            all_wmape = df_combined['wmape'].values
            all_wmape = all_wmape[np.isfinite(all_wmape)]
            
            valid_acc = ~(pd.isna(df_combined['real_acc']) | pd.isna(df_combined['pred_acc'])) & (df_combined['real_acc'] > 0)
            mape_acc = 100 * np.abs(
                df_combined.loc[valid_acc, 'real_acc'] - df_combined.loc[valid_acc, 'pred_acc']
            ) / df_combined.loc[valid_acc, 'real_acc']
            mape_acc = mape_acc[np.isfinite(mape_acc)]
            
            valid_daily = ~(pd.isna(df_combined['real_daily_avg']) | pd.isna(df_combined['pred_daily_avg'])) & (df_combined['real_daily_avg'] > 0)
            mape_daily = 100 * np.abs(
                df_combined.loc[valid_daily, 'real_daily_avg'] - df_combined.loc[valid_daily, 'pred_daily_avg']
            ) / df_combined.loc[valid_daily, 'real_daily_avg']
            mape_daily = mape_daily[np.isfinite(mape_daily)]
            
            if len(all_wmape) > 0:
                wmape_stats = MetricStats.from_array(all_wmape)
                print(f"  WMAPE: Mediana={wmape_stats.median:.4f}, P90={wmape_stats.p90:.4f}")
            
            if len(mape_acc) > 0:
                mape_acc_stats = MetricStats.from_array(mape_acc)
                print(f"  MAPE Acumulado: Mediana={mape_acc_stats.median:.2f}%, P90={mape_acc_stats.p90:.2f}%")
            
            if len(mape_daily) > 0:
                mape_daily_stats = MetricStats.from_array(mape_daily)
                print(f"  MAPE Diário: Mediana={mape_daily_stats.median:.2f}%, P90={mape_daily_stats.p90:.2f}%")
        
        # AGREGADO GLOBAL (todas classes e history_sizes)
        print("\n" + "=" * 80)
        print("AGREGADO GLOBAL (todas as classes e history_sizes)")
        print("=" * 80)
        
        all_dfs = []
        for hist_metrics in self.metrics.values():
            for df in hist_metrics.values():
                all_dfs.append(df)
        
        if all_dfs:
            df_global = pd.concat(all_dfs, ignore_index=True)
            
            all_wmape = df_global['wmape'].values
            all_wmape = all_wmape[np.isfinite(all_wmape)]
            
            valid_acc = ~(pd.isna(df_global['real_acc']) | pd.isna(df_global['pred_acc'])) & (df_global['real_acc'] > 0)
            mape_acc_global = 100 * np.abs(
                df_global.loc[valid_acc, 'real_acc'] - df_global.loc[valid_acc, 'pred_acc']
            ) / df_global.loc[valid_acc, 'real_acc']
            mape_acc_global = mape_acc_global[np.isfinite(mape_acc_global)]
            
            valid_daily = ~(pd.isna(df_global['real_daily_avg']) | pd.isna(df_global['pred_daily_avg'])) & (df_global['real_daily_avg'] > 0)
            mape_daily_global = 100 * np.abs(
                df_global.loc[valid_daily, 'real_daily_avg'] - df_global.loc[valid_daily, 'pred_daily_avg']
            ) / df_global.loc[valid_daily, 'real_daily_avg']
            mape_daily_global = mape_daily_global[np.isfinite(mape_daily_global)]
            
            if len(all_wmape) > 0:
                wmape_global = MetricStats.from_array(all_wmape)
                print(f"  WMAPE: Mediana={wmape_global.median:.4f}, P90={wmape_global.p90:.4f}")
            
            if len(mape_acc_global) > 0:
                mape_acc_stats = MetricStats.from_array(mape_acc_global)
                print(f"  MAPE Acumulado: Mediana={mape_acc_stats.median:.2f}%, P90={mape_acc_stats.p90:.2f}%")
            
            if len(mape_daily_global) > 0:
                mape_daily_stats = MetricStats.from_array(mape_daily_global)
                print(f"  MAPE Diário: Mediana={mape_daily_stats.median:.2f}%, P90={mape_daily_stats.p90:.2f}%")
            
            print()
        
        print("=" * 80)
        print()
    
    
    def compare_history_sizes(
        self,
        class_id: Optional[int] = None,
        metric: str = 'mape_acc',
        figsize: tuple = (14, 6)
    ):
        """
        Compara performance entre diferentes history_sizes
        
        Parameters:
        -----------
        class_id : int, optional
            Se especificado, compara apenas nesta classe. Se None, agrega todas
        metric : str, optional
            Métrica: 'wmape', 'mape_acc', 'mape_daily', etc.
        figsize : tuple
            Tamanho da figura
        """
        _, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Coletar dados
        hist_data = {}
        
        for _, hist_metrics in self.metrics.items():
            for history_size, df_metrics in hist_metrics.items():
                ds_class_id = df_metrics['class'].iloc[0]
                
                # Filtrar por classe se especificado
                if class_id is not None and ds_class_id != class_id:
                    continue
                
                # Extrair métrica
                if metric == 'mape_acc':
                    valid = ~(pd.isna(df_metrics['real_acc']) | pd.isna(df_metrics['pred_acc'])) & (df_metrics['real_acc'] > 0)
                    if valid.any():
                        values = 100 * np.abs(
                            df_metrics.loc[valid, 'real_acc'] - df_metrics.loc[valid, 'pred_acc']
                        ) / df_metrics.loc[valid, 'real_acc']
                        values = values[np.isfinite(values)]
                
                elif metric == 'mape_daily':
                    valid = ~(pd.isna(df_metrics['real_daily_avg']) | pd.isna(df_metrics['pred_daily_avg'])) & (df_metrics['real_daily_avg'] > 0)
                    if valid.any():
                        values = 100 * np.abs(
                            df_metrics.loc[valid, 'real_daily_avg'] - df_metrics.loc[valid, 'pred_daily_avg']
                        ) / df_metrics.loc[valid, 'real_daily_avg']
                        values = values[np.isfinite(values)]
                
                else:
                    values = df_metrics[metric].values
                    values = values[np.isfinite(values)]
                
                label = f"Hist {history_size}"
                
                if label not in hist_data:
                    hist_data[label] = []
                
                if len(values) > 0:
                    hist_data[label].extend(values)
        
        # Converter para arrays
        hist_data = {k: np.array(v) for k, v in hist_data.items() if len(v) > 0}
        
        if not hist_data:
            print("Nenhum dado válido")
            return
        
        # Boxplot
        ax1 = axes[0]
        
        # Ordenar por history_size
        sorted_items = sorted(hist_data.items(), key=lambda x: int(x[0].split()[1]))
        labels_sorted = [item[0] for item in sorted_items]
        values_sorted = [item[1] for item in sorted_items]
        
        bp = ax1.boxplot(
            values_sorted,
            tick_labels=labels_sorted,
            patch_artist=True,
            showfliers=False
        )
        
        colors = plt.cm.plasma(np.linspace(0, 1, len(labels_sorted)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_ylabel(metric.upper(), fontsize=12, fontweight='bold')
        ax1.set_title(f'Distribuição por History Size', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Ranking
        ax2 = axes[1]
        medians = [np.median(v) for v in values_sorted]
        
        bars = ax2.barh(range(len(labels_sorted)), medians, color=colors, alpha=0.8)
        
        ax2.set_yticks(range(len(labels_sorted)))
        ax2.set_yticklabels(labels_sorted)
        ax2.set_xlabel(f'{metric.upper()} (Mediana)', fontsize=12, fontweight='bold')
        ax2.set_title('Ranking por History Size', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        for i, (bar, val) in enumerate(zip(bars, medians)):
            ax2.text(val, i, f' {val:.2f}', va='center', fontsize=10, fontweight='bold')
        
        class_str = f"Classe {class_id}" if class_id is not None else "Todas as classes"
        plt.suptitle(f'BASELINE | {class_str} | {metric.upper()} | Target: {self.target}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    
    def compare_classes_by_history(
        self,
        history_size: int,
        metric: str = 'mape_acc',
        figsize: tuple = (14, 6)
    ):
        """
        Compara classes para um history_size específico
        
        Parameters:
        -----------
        history_size : int
            Tamanho do histórico
        metric : str
            Métrica a comparar
        figsize : tuple
            Tamanho da figura
        """
        _, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Coletar dados
        class_data = {}
        
        for _, hist_metrics in self.metrics.items():
            if history_size not in hist_metrics:
                continue
            
            df_metrics = hist_metrics[history_size]
            class_id = df_metrics['class'].iloc[0]
            
            # Extrair métrica
            if metric == 'mape_acc':
                valid = ~(pd.isna(df_metrics['real_acc']) | pd.isna(df_metrics['pred_acc'])) & (df_metrics['real_acc'] > 0)
                if valid.any():
                    values = 100 * np.abs(
                        df_metrics.loc[valid, 'real_acc'] - df_metrics.loc[valid, 'pred_acc']
                    ) / df_metrics.loc[valid, 'real_acc']
                    values = values[np.isfinite(values)]
            
            elif metric == 'mape_daily':
                valid = ~(pd.isna(df_metrics['real_daily_avg']) | pd.isna(df_metrics['pred_daily_avg'])) & (df_metrics['real_daily_avg'] > 0)
                if valid.any():
                    values = 100 * np.abs(
                        df_metrics.loc[valid, 'real_daily_avg'] - df_metrics.loc[valid, 'pred_daily_avg']
                    ) / df_metrics.loc[valid, 'real_daily_avg']
                    values = values[np.isfinite(values)]
            
            else:
                values = df_metrics[metric].values
                values = values[np.isfinite(values)]
            
            if len(values) > 0:
                class_data[f"Classe {class_id}"] = np.array(values)
        
        if not class_data:
            print(f"Nenhum dado para history_size={history_size}")
            return
        
        # Boxplot
        ax1 = axes[0]
        bp = ax1.boxplot(
            list(class_data.values()),
            tick_labels=list(class_data.keys()),
            patch_artist=True,
            showfliers=False
        )
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(class_data)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_ylabel(metric.upper(), fontsize=12, fontweight='bold')
        ax1.set_title(f'Distribuição por Classe', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Ranking
        ax2 = axes[1]
        class_names = list(class_data.keys())
        medians = [np.median(v) for v in class_data.values()]
        
        sorted_idx = np.argsort(medians)
        class_names_sorted = [class_names[i] for i in sorted_idx]
        medians_sorted = [medians[i] for i in sorted_idx]
        colors_sorted = [colors[i] for i in sorted_idx]
        
        bars = ax2.barh(range(len(class_names_sorted)), medians_sorted, color=colors_sorted, alpha=0.8)
        
        ax2.set_yticks(range(len(class_names_sorted)))
        ax2.set_yticklabels(class_names_sorted)
        ax2.set_xlabel(f'{metric.upper()} (Mediana)', fontsize=12, fontweight='bold')
        ax2.set_title('Ranking por Classe', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        for i, (bar, val) in enumerate(zip(bars, medians_sorted)):
            ax2.text(val, i, f' {val:.2f}', va='center', fontsize=10, fontweight='bold')
        
        plt.suptitle(f'BASELINE | History {history_size} | {metric.upper()} | Target: {self.target}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    
    def compare_heatmap(
        self,
        metric: str = 'mape_acc',
        figsize: tuple = (10, 6)
    ):
        """
        Heatmap comparando classes × history_sizes
        
        Parameters:
        -----------
        metric : str
            Métrica a visualizar
        figsize : tuple
            Tamanho da figura
        """
        # Coletar dados em matriz
        data_matrix = {}
        
        for _, hist_metrics in self.metrics.items():
            for history_size, df_metrics in hist_metrics.items():
                class_id = df_metrics['class'].iloc[0]
                
                # Extrair métrica
                if metric == 'mape_acc':
                    valid = ~(pd.isna(df_metrics['real_acc']) | pd.isna(df_metrics['pred_acc'])) & (df_metrics['real_acc'] > 0)
                    if valid.any():
                        values = 100 * np.abs(
                            df_metrics.loc[valid, 'real_acc'] - df_metrics.loc[valid, 'pred_acc']
                        ) / df_metrics.loc[valid, 'real_acc']
                        median_val = np.median(values[np.isfinite(values)])
                
                elif metric == 'mape_daily':
                    valid = ~(pd.isna(df_metrics['real_daily_avg']) | pd.isna(df_metrics['pred_daily_avg'])) & (df_metrics['real_daily_avg'] > 0)
                    if valid.any():
                        values = 100 * np.abs(
                            df_metrics.loc[valid, 'real_daily_avg'] - df_metrics.loc[valid, 'pred_daily_avg']
                        ) / df_metrics.loc[valid, 'real_daily_avg']
                        median_val = np.median(values[np.isfinite(values)])
                
                else:
                    values = df_metrics[metric].values
                    values = values[np.isfinite(values)]
                    median_val = np.median(values) if len(values) > 0 else np.nan
                
                if class_id not in data_matrix:
                    data_matrix[class_id] = {}
                
                data_matrix[class_id][history_size] = median_val
        
        # Criar DataFrame para heatmap
        df_heatmap = pd.DataFrame(data_matrix).T
        df_heatmap = df_heatmap.sort_index(axis=0).sort_index(axis=1)
        
        # Plotar
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(df_heatmap.values, cmap='RdYlGn_r', aspect='auto')
        
        # Eixos
        ax.set_xticks(np.arange(len(df_heatmap.columns)))
        ax.set_yticks(np.arange(len(df_heatmap.index)))
        ax.set_xticklabels([f'Hist {h}' for h in df_heatmap.columns])
        ax.set_yticklabels([f'Classe {c}' for c in df_heatmap.index])
        
        # Valores nas células
        for i in range(len(df_heatmap.index)):
            for j in range(len(df_heatmap.columns)):
                val = df_heatmap.values[i, j]
                if not np.isnan(val):
                    text = ax.text(j, i, f'{val:.1f}',
                                 ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title(f'BASELINE | {metric.upper()} (Mediana) | Target: {self.target}', fontsize=14, fontweight='bold')
        ax.set_xlabel('History Size', fontsize=12, fontweight='bold')
        ax.set_ylabel('Classe', fontsize=12, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(metric.upper(), rotation=270, labelpad=20, fontweight='bold')
        
        plt.tight_layout()
        plt.show()    
    def compare_classes(
        self,
        metric: str = 'mape_acc',
        figsize: tuple = (14, 6)
    ):
        """Compara performance entre classes"""
        _, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Agrupar por classe
        class_data = {}
        
        for _, hist_metrics in self.metrics.items():
            for _, df_metrics in hist_metrics.items():
                class_id = df_metrics['class'].iloc[0]
                
                if metric == 'mape_acc':
                    valid = ~(pd.isna(df_metrics['real_acc']) | pd.isna(df_metrics['pred_acc'])) & (df_metrics['real_acc'] > 0)
                    if valid.any():
                        values = 100 * np.abs(
                            df_metrics.loc[valid, 'real_acc'] - df_metrics.loc[valid, 'pred_acc']
                        ) / df_metrics.loc[valid, 'real_acc']
                        values = values[np.isfinite(values)]
                
                elif metric == 'mape_daily':
                    valid = ~(pd.isna(df_metrics['real_daily_avg']) | pd.isna(df_metrics['pred_daily_avg'])) & (df_metrics['real_daily_avg'] > 0)
                    if valid.any():
                        values = 100 * np.abs(
                            df_metrics.loc[valid, 'real_daily_avg'] - df_metrics.loc[valid, 'pred_daily_avg']
                        ) / df_metrics.loc[valid, 'real_daily_avg']
                        values = values[np.isfinite(values)]
                
                else:
                    values = df_metrics[metric].values
                    values = values[np.isfinite(values)]
                
                if f"Classe {class_id}" not in class_data:
                    class_data[f"Classe {class_id}"] = []
                
                if len(values) > 0:
                    class_data[f"Classe {class_id}"].extend(values)
        
        # Converter para arrays
        class_data = {k: np.array(v) for k, v in class_data.items() if len(v) > 0}
        
        if not class_data:
            print("Nenhum dado válido")
            return
        
        # Boxplot
        ax1 = axes[0]
        bp = ax1.boxplot(
            list(class_data.values()),
            tick_labels=list(class_data.keys()),
            patch_artist=True,
            showfliers=False
        )
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(class_data)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_ylabel(metric.upper(), fontsize=12, fontweight='bold')
        ax1.set_title(f'Distribuição por Classe', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Ranking
        ax2 = axes[1]
        class_names = list(class_data.keys())
        medians = [np.median(v) for v in class_data.values()]
        
        sorted_idx = np.argsort(medians)
        class_names_sorted = [class_names[i] for i in sorted_idx]
        medians_sorted = [medians[i] for i in sorted_idx]
        colors_sorted = [colors[i] for i in sorted_idx]
        
        bars = ax2.barh(range(len(class_names_sorted)), medians_sorted, color=colors_sorted, alpha=0.8)
        
        ax2.set_yticks(range(len(class_names_sorted)))
        ax2.set_yticklabels(class_names_sorted)
        ax2.set_xlabel(f'{metric.upper()} (Mediana)', fontsize=12, fontweight='bold')
        ax2.set_title('Ranking por Classe', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        for i, (bar, val) in enumerate(zip(bars, medians_sorted)):
            ax2.text(val, i, f' {val:.2f}', va='center', fontsize=10, fontweight='bold')
        
        plt.suptitle(f'BASELINE | {metric.upper()} | Target: {self.target}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def export_results(self, output_dir: Optional[str] = None) -> Dict[str, Path]:
        """Exporta resultados"""
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = Path('.')
        
        exported = {}
        
        print("💾 Exportando resultados do baseline...")
        
        for (dataset_name, class_id), hist_metrics in self.metrics.items():
            for history_size, df_metrics in hist_metrics.items():
                
                # Calcular MAPE
                df_copy = df_metrics.copy()
                
                valid_acc = ~(pd.isna(df_copy['real_acc']) | pd.isna(df_copy['pred_acc'])) & (df_copy['real_acc'] > 0)
                df_copy['mape_acc'] = np.nan
                if valid_acc.any():
                    df_copy.loc[valid_acc, 'mape_acc'] = 100 * np.abs(
                        df_copy.loc[valid_acc, 'real_acc'] - df_copy.loc[valid_acc, 'pred_acc']
                    ) / df_copy.loc[valid_acc, 'real_acc']
                
                valid_daily = ~(pd.isna(df_copy['real_daily_avg']) | pd.isna(df_copy['pred_daily_avg'])) & (df_copy['real_daily_avg'] > 0)
                df_copy['mape_daily'] = np.nan
                if valid_daily.any():
                    df_copy.loc[valid_daily, 'mape_daily'] = 100 * np.abs(
                        df_copy.loc[valid_daily, 'real_daily_avg'] - df_copy.loc[valid_daily, 'pred_daily_avg']
                    ) / df_copy.loc[valid_daily, 'real_daily_avg']
                
                df_copy['model'] = 'LinearTrendBaseline'
                df_copy['history_size'] = history_size
                
                filename = f'baseline_class{class_id}_hist{history_size}_{self.target}.csv'
                filepath = output_path / filename
                
                df_copy.to_csv(filepath, index=False)
                exported[f"{dataset_name}_class{class_id}"] = filepath
                
                if self.verbose:
                    file_size = filepath.stat().st_size / 1024
                    print(f"  ✓ Classe {class_id} (hist={history_size}): {file_size:.1f} KB")
        
        if self.verbose:
            print()
        
        return exported
    
    def __repr__(self) -> str:
        return (
            f"BaselineEvaluator(\n"
            f"  target='{self.target}',\n"
            f"  n_datasets={len(self.local_datasets)},\n"
            f"  baseline={self.baseline_model.__class__.__name__}\n"
            f")"
        )


class GlobalModelEvaluator:
    """
    Avaliador completo de modelos em múltiplos datasets (classes) de teste
    
    Gerencia predições e avaliação com cache automático, suportando:
    - Auto-descoberta de datasets por target
    - Cache de predições para evitar recálculo
    - Modelos PyTorch e Sklearn
    - Modelos com likelihood (quantis) e sem
    - Desnormalização via VehicleProfile
    - Análise de performance diária e acumulada
    - Comparação entre classes e modelos
    """
    
    def __init__(
        self,
        models_dir: str,
        datasets_dir: str,
        vp: VehicleProfile,
        likelihood: bool = True,
        model_names: Optional[List[str]] = None,
        cache_dir: str = '.cache/predictions',
        n_jobs: int = -1,
        verbose: bool = True
    ):
        """
        Inicializa o avaliador
        
        Parameters:
        -----------
        models_dir : str
            Diretório raiz dos modelos
        datasets_dir : str
            Diretório com datasets de teste
        vp : VehicleProfile
            Perfil de veículos
        likelihood : bool, optional
            Se True, modelos usam quantis (padrão: True)
        model_names : List[str], optional
            Modelos a avaliar. Se None, avalia todos
        cache_dir : str, optional
            Diretório para cache de predições (padrão: '.cache/predictions')
        n_jobs : int, optional
            Jobs paralelos (padrão: -1)
        verbose : bool, optional
            Logs detalhados (padrão: True)
        """
        self.models_dir = Path(models_dir)
        self.datasets_dir = Path(datasets_dir)
        self.vp = vp
        self.likelihood = likelihood
        self.cache_dir = Path(cache_dir)
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # Criar diretório de cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Extrair target
        self.target = vp.target
        
        if self.verbose:
            print("=" * 80)
            print("🔮 INICIALIZANDO AVALIAÇÃO COMPLETA")
            print("=" * 80)
            print(f"Models dir: {self.models_dir}")
            print(f"Datasets dir: {self.datasets_dir}")
            print(f"Cache dir: {self.cache_dir}")
            print(f"Target: {self.target}")
            print(f"Likelihood: {likelihood}")
            print()
        
        # Descobrir datasets
        self.test_datasets = self._discover_test_datasets()
        
        if not self.test_datasets:
            raise ValueError(f"Nenhum dataset encontrado para target '{self.target}'")
        
        if self.verbose:
            print(f"✓ {len(self.test_datasets)} dataset(s) encontrado(s):")
            for dataset_info in self.test_datasets:
                print(f"  • {dataset_info['name']} (classe {dataset_info['class_id']})")
            print()
        
        # Carregar dados de teste
        self.loaders = {}
        self.test_windows = {}
        
        for dataset_info in self.test_datasets:
            dataset_name = dataset_info['name']
            
            if self.verbose:
                print(f"Carregando {dataset_name}...")
            
            loader = DatasetLoader(
                dataset_dir=self.datasets_dir / dataset_name,
                vp=vp,
                verbose=False
            )
            
            windows = loader.load(
                split='test',
                return_windows=True,
                use_cache=True,
                cache_dir=str(self.cache_dir / 'windows')
            )
            
            self.loaders[dataset_name] = loader
            self.test_windows[dataset_name] = windows
            
            if self.verbose:
                n_series = sum(len(w) for w in windows)
                print(f"  ✓ {len(windows)} janelas, {n_series} séries")
        
        if self.verbose:
            print()
        
        # Descobrir modelos
        self.available_models = self._discover_models()
        
        if model_names:
            self.model_names = [m for m in model_names if m in self.available_models]
        else:
            self.model_names = list(self.available_models.keys())
        
        if self.verbose:
            print(f"Modelos disponíveis: {list(self.available_models.keys())}")
            print(f"Modelos a avaliar: {self.model_names}")
            print("=" * 80)
            print()
        
        # Resultados
        self.predictions = {}
        self.metrics = {}
    
    def _discover_test_datasets(self) -> List[Dict[str, Any]]:
        """Descobre datasets de teste do target"""
        datasets = []
        
        pattern = f"dataset_{self.target}_*"
        
        for dataset_path in self.datasets_dir.glob(pattern):
            if not dataset_path.is_dir():
                continue
            
            dataset_name = dataset_path.name
            
            # Extrair class_id
            match = re.search(r'class(\d+)', dataset_name)
            if match:
                class_id = int(match.group(1))
            else:
                class_id = None
            
            datasets.append({
                'name': dataset_name,
                'class_id': class_id,
                'path': dataset_path
            })
        
        datasets.sort(key=lambda x: x['class_id'] if x['class_id'] is not None else -1)
        
        return datasets
    
    def _discover_models(self) -> Dict[str, Dict[str, Any]]:
        """Descobre modelos para cada dataset"""
        models = {}
        
        for model_dir in self.models_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            model_name = model_dir.name
            models[model_name] = {}
            
            for dataset_info in self.test_datasets:
                dataset_name = dataset_info['name']
                dataset_subdir = model_dir / dataset_name
                
                if not dataset_subdir.exists():
                    continue
                
                # Buscar checkpoints
                checkpoints_torch = list(dataset_subdir.glob('model_*/checkpoints/*.ckpt'))
                checkpoints_pkl = list(dataset_subdir.glob('model_*.pkl'))
                
                if not checkpoints_torch and not checkpoints_pkl:
                    continue
                
                is_torch = len(checkpoints_torch) > 0
                
                checkpoint_indices = []
                if is_torch:
                    for cp in checkpoints_torch:
                        idx = int(cp.parent.parent.name.split('_')[1])
                        checkpoint_indices.append(idx)
                else:
                    for cp in checkpoints_pkl:
                        idx = int(cp.stem.split('_')[1])
                        checkpoint_indices.append(idx)
                
                checkpoint_indices = sorted(set(checkpoint_indices))
                
                models[model_name][dataset_name] = {
                    'work_dir': dataset_subdir,
                    'is_torch': is_torch,
                    'checkpoints': checkpoint_indices
                }
        
        models = {k: v for k, v in models.items() if v}
        
        return models
    
    def _compute_predictions_cache_hash(
        self,
        model_name: str,
        dataset_name: str,
        checkpoint_idx: int
    ) -> str:
        """
        Computa hash para cache de predições
        
        Baseado em:
        - Caminho do modelo
        - Timestamp do checkpoint
        - Configuração do dataset
        """
        model_info = self.available_models[model_name][dataset_name]
        work_dir = model_info['work_dir']
        is_torch = model_info['is_torch']
        
        # Obter timestamp do checkpoint
        if is_torch:
            checkpoint_path = work_dir / f'model_{checkpoint_idx:02d}'
            if checkpoint_path.exists():
                checkpoint_time = checkpoint_path.stat().st_mtime
            else:
                checkpoint_time = 0
        else:
            checkpoint_path = work_dir / f'model_{checkpoint_idx:02d}.pkl'
            if checkpoint_path.exists():
                checkpoint_time = checkpoint_path.stat().st_mtime
            else:
                checkpoint_time = 0
        
        # Criar string única
        hash_str = (
            f"{model_name}_{dataset_name}_{checkpoint_idx}_"
            f"{checkpoint_time}_{self.likelihood}"
        )
        
        return hashlib.md5(hash_str.encode()).hexdigest()
    
    def _try_load_predictions_from_cache(
        self,
        model_name: str,
        dataset_name: str,
        checkpoint_idx: int
    ) -> Optional[Dict[str, Any]]:
        """
        Tenta carregar predições do cache
        
        Returns:
        --------
        dict or None
            {'predictions': [...], 'quantiles': [...]} ou None se não encontrado
        """
        cache_hash = self._compute_predictions_cache_hash(model_name, dataset_name, checkpoint_idx)
        cache_file = self.cache_dir / f'pred_{cache_hash}.pkl'
        
        if not cache_file.exists():
            return None
        
        try:
            if self.verbose:
                print(f"    📦 Carregando predições do cache...")
            
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            if self.verbose:
                print(f"    ✓ Cache carregado: {cache_file.name}")
            
            return cached_data
        
        except Exception as e:
            warnings.warn(f"Erro ao carregar cache: {e}. Recalculando...")
            return None
    
    def _save_predictions_to_cache(
        self,
        model_name: str,
        dataset_name: str,
        checkpoint_idx: int,
        predictions_data: Dict[str, Any]
    ):
        """
        Salva predições no cache
        
        Parameters:
        -----------
        model_name : str
            Nome do modelo
        dataset_name : str
            Nome do dataset
        checkpoint_idx : int
            Índice do checkpoint
        predictions_data : dict
            {'predictions': [...], 'quantiles': [...]}
        """
        cache_hash = self._compute_predictions_cache_hash(model_name, dataset_name, checkpoint_idx)
        cache_file = self.cache_dir / f'pred_{cache_hash}.pkl'
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(predictions_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            if self.verbose:
                file_size = cache_file.stat().st_size / (1024 * 1024)
                print(f"    💾 Cache salvo: {cache_file.name} ({file_size:.2f} MB)")
        
        except Exception as e:
            warnings.warn(f"Erro ao salvar cache: {e}")
    
    def _load_model(self, model_name: str, dataset_name: str, checkpoint_idx: int = 0):
        """Carrega modelo com fallback robusto"""
        model_info = self.available_models[model_name][dataset_name]
        work_dir = model_info['work_dir']
        is_torch = model_info['is_torch']
        
        model_classes = {
            'DLinearModel': DLinearModel,
            'NLinearModel': NLinearModel,
            'TSMixerModel': TSMixerModel,
            'TCNModel': TCNModel,
            'NBEATSModel': NBEATSModel,
            'NHiTSModel': NHiTSModel,
            'LightGBMModel': LightGBMModel,
            'XGBModel': XGBModel,
            'LinearRegressionModel': LinearRegressionModel
        }
        
        model_cls = model_classes.get(model_name)
        
        if model_cls is None:
            raise ValueError(f"Modelo {model_name} não suportado")
        
        if not is_torch:
            model_path = work_dir / f'model_{checkpoint_idx:02d}.pkl'
            return model_cls.load(str(model_path))
        
        # PyTorch: tentar estratégias
        try:
            model = model_cls.load_from_checkpoint(
                model_name=f'model_{checkpoint_idx:02d}',
                work_dir=str(work_dir),
                best=True
            )
            return model
        
        except Exception as e1:
            # Fallback: carregar manualmente
            try:
                model_dir = work_dir / f'model_{checkpoint_idx:02d}'
                pth_file = model_dir / '_model.pth.tar'
                
                if not pth_file.exists():
                    raise FileNotFoundError(f"Arquivo não encontrado: {pth_file}")
                
                checkpoint = torch.load(pth_file, map_location='cpu')
                
                if 'hyper_parameters' in checkpoint:
                    hparams = checkpoint['hyper_parameters']
                    hparams_clean = {k: v for k, v in hparams.items() 
                                    if k not in ['work_dir', 'model_name', 'pl_trainer_kwargs']}
                    
                    model = model_cls(**hparams_clean)
                    
                    if 'state_dict' in checkpoint:
                        model.model.load_state_dict(checkpoint['state_dict'])
                    
                    return model
                else:
                    raise KeyError("Checkpoint sem 'hyper_parameters'")
            
            except Exception as e2:
                raise RuntimeError(
                    f"Erro ao carregar {model_name}/{dataset_name}:\n"
                    f"  1. load_from_checkpoint: {e1}\n"
                    f"  2. Manual load: {e2}"
                )
    
    def _get_quantiles(self, model) -> Optional[List[float]]:
        """Extrai quantis do modelo"""
        if not self.likelihood:
            return None
        
        if not hasattr(model, 'likelihood') or model.likelihood is None:
            return None
        
        if hasattr(model.likelihood, 'quantiles'):
            return model.likelihood.quantiles
        
        return None
    
    def predict_all(self, use_cache: bool = True, force_reload: bool = False):
        """Gera predições para todos os modelos em todos os datasets"""
        if self.verbose:
            print("=" * 80)
            print("🔮 GERANDO PREDIÇÕES")
            print("=" * 80)
            print()
        
        for model_name in self.model_names:
            if self.verbose:
                print(f"\nModelo: {model_name}")
                print("-" * 80)
            
            self.predictions[model_name] = {}
            
            for dataset_info in self.test_datasets:
                dataset_name = dataset_info['name']
                class_id = dataset_info['class_id']
                
                if dataset_name not in self.available_models.get(model_name, {}):
                    if self.verbose:
                        print(f"  ⚠️  Classe {class_id}: modelo não encontrado")
                    continue
                
                if self.verbose:
                    print(f"  Classe {class_id}:")
                
                # Tentar carregar do cache
                checkpoint_idx = 0
                cached_pred = None
                
                if use_cache and not force_reload:
                    cached_pred = self._try_load_predictions_from_cache(
                        model_name, dataset_name, checkpoint_idx
                    )
                
                if cached_pred is not None:
                    self.predictions[model_name][dataset_name] = cached_pred
                    
                    if self.verbose:
                        print(f"    ✓ Classe {class_id} carregada do cache")
                    
                    continue
                
                # Gerar predições
                try:
                    model = self._load_model(model_name, dataset_name, checkpoint_idx)
                    quantiles = self._get_quantiles(model)
                    
                    if quantiles and self.verbose:
                        print(f"    Quantis: {quantiles}")
                    
                    windows = self.test_windows[dataset_name]
                    dataset_predictions = []
                    
                    for idx, window in enumerate(windows):
                        if self.verbose:
                            print(f"    Janela {idx + 1}/{len(windows)}: {len(window)} séries", end='')
                        
                        # Argumentos base
                        predict_args = {
                            'n': self.loaders[dataset_name].forecast_horizon,
                            'series': window.input_series,
                        }
                        
                        # Adicionar covariáveis se suportado
                        if model.supports_past_covariates and window.input_past_covariates is not None:
                            predict_args['past_covariates'] = window.input_past_covariates
                        
                        if model.supports_future_covariates and window.output_future_covariates is not None:
                            predict_args['future_covariates'] = window.output_future_covariates
                        
                        if isinstance(model, TorchForecastingModel):
                            predict_args['n_jobs'] = self.n_jobs
                        
                        # Adicionar parâmetros de likelihood
                        if quantiles:
                            predict_args['num_samples'] = 1
                            predict_args['predict_likelihood_parameters'] = True
                        
                        predictions = model.predict(**predict_args)
                        dataset_predictions.append(predictions)
                        
                        if self.verbose:
                            print(" ✓")
                    
                    predictions_data = {
                        'predictions': dataset_predictions,
                        'quantiles': quantiles,
                        'class_id': class_id
                    }
                    
                    self.predictions[model_name][dataset_name] = predictions_data
                    
                    # Salvar no cache
                    if use_cache:
                        self._save_predictions_to_cache(
                            model_name, dataset_name, checkpoint_idx, predictions_data
                        )
                    
                    if self.verbose:
                        print(f"    ✓ Classe {class_id} concluída")
                
                except Exception as e:
                    print(f"    ❌ Erro: {e}")
        
        if self.verbose:
            print("\n" + "=" * 80)
            print()
        
    def calculate_metrics(self):
        """Calcula métricas - VERSÃO CORRIGIDA FINAL"""
        if self.verbose:
            print("=" * 80)
            print("📊 CALCULANDO MÉTRICAS")
            print("=" * 80)
            print()
        
        for model_name, datasets_pred in self.predictions.items():
            if self.verbose:
                print(f"\nModelo: {model_name}")
                print("-" * 80)
            
            self.metrics[model_name] = {}
            
            for dataset_name, pred_data in datasets_pred.items():
                class_id = pred_data['class_id']
                
                if self.verbose:
                    print(f"  Classe {class_id}:")
                
                model_predictions = pred_data['predictions']
                quantiles = pred_data['quantiles']
                windows = self.test_windows[dataset_name]
                
                dataset_metrics = []
                
                for idx, (window, predictions) in enumerate(zip(windows, model_predictions)):
                    if self.verbose:
                        print(f"    Janela {idx + 1}/{len(windows)}", end='')
                    
                    metrics_data = {
                        'veiculo_id': window.ids,
                        'window': idx,
                        'class': class_id
                    }
                    
                    # Verificar estrutura da predição
                    if len(predictions) > 0:
                        pred_shape = predictions[0].all_values(copy=False).shape
                        # pred_shape = (n_timesteps, n_quantiles, n_samples)
                        # Exemplo: (7, 7, 1)
                        
                        n_timesteps = pred_shape[0]
                        n_components_or_quantiles = pred_shape[1]
                        n_samples = pred_shape[2] if len(pred_shape) > 2 else 1
                        
                        # Se n_components == len(quantiles), são quantis!
                        if quantiles and n_components_or_quantiles == len(quantiles):
                            # Cada "componente" é na verdade um quantil
                            for q_idx, q in enumerate(quantiles):
                                # Extrair quantil: [:, q_idx, 0]
                                # Shape resultante: (7, 1)
                                pred_q = [
                                    TimeSeries.from_times_and_values(
                                        times=pred.time_index,
                                        values=pred.all_values(copy=False)[:, q_idx:q_idx+1, 0],  # ← CORRIGIDO
                                        freq=pred.freq
                                    )
                                    for pred in predictions
                                ]
                                
                                metrics_data[f'wmape_q{int(q*100):02d}'] = wmape(window.output_series, pred_q)
                                metrics_data[f'mae_q{int(q*100):02d}'] = mae(window.output_series, pred_q)
                            
                            # Usar mediana (q=0.5)
                            median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
                            
                            pred_main = [
                                TimeSeries.from_times_and_values(
                                    times=pred.time_index,
                                    values=pred.all_values(copy=False)[:, median_idx:median_idx+1, 0],  # ← CORRIGIDO
                                    freq=pred.freq
                                )
                                for pred in predictions
                            ]
                        else:
                            # Não tem quantis ou estrutura diferente
                            pred_main = predictions
                    else:
                        pred_main = predictions
                    
                    # Métricas principais
                    metrics_data['wmape'] = wmape(window.output_series, pred_main)
                    metrics_data['smape'] = smape(window.output_series, pred_main)
                    metrics_data['r2'] = r2_score(window.output_series, pred_main)
                    metrics_data['mae'] = mae(window.output_series, pred_main)
                    metrics_data['mse'] = mse(window.output_series, pred_main)
                    
                    # Acumulados (desnormalizados)
                    real_acc = []
                    pred_acc = []
                    real_daily_avg = []
                    pred_daily_avg = []
                    
                    for i, vid in enumerate(window.ids):
                        history_end_date = window.input_series[i].end_time()
                        
                        try:
                            real_values = self.vp.inverse_transform(
                                vehicle_id=vid,
                                scaled_values=window.output_series[i].values().flatten(),
                                date=history_end_date
                            )
                            
                            pred_values = self.vp.inverse_transform(
                                vehicle_id=vid,
                                scaled_values=pred_main[i].values().flatten(),
                                date=history_end_date
                            )
                            
                            real_acc.append(real_values.sum())
                            pred_acc.append(pred_values.sum())
                            real_daily_avg.append(real_values.mean())
                            pred_daily_avg.append(pred_values.mean())
                        
                        except Exception as e:
                            warnings.warn(f"Erro ao desnormalizar veículo {vid}: {e}")
                            real_acc.append(np.nan)
                            pred_acc.append(np.nan)
                            real_daily_avg.append(np.nan)
                            pred_daily_avg.append(np.nan)
                    
                    metrics_data['real_acc'] = real_acc
                    metrics_data['pred_acc'] = pred_acc
                    metrics_data['real_daily_avg'] = real_daily_avg
                    metrics_data['pred_daily_avg'] = pred_daily_avg
                    
                    df_metrics = pd.DataFrame(metrics_data)
                    dataset_metrics.append(df_metrics)
                    
                    if self.verbose:
                        print(" ✓")
                
                self.metrics[model_name][dataset_name] = dataset_metrics
                
                if self.verbose:
                    print(f"    ✓ Classe {class_id} concluída")
        
            if self.verbose:
                print("\n" + "=" * 80)
                print()
                self.print_summary()    
    def print_summary(self):
        """Imprime resumo completo"""
        print("=" * 80)
        print("📊 RESUMO COMPLETO DAS MÉTRICAS")
        print("=" * 80)
        print()
        
        for model_name in self.model_names:
            if model_name not in self.metrics:
                continue
            
            print(f"\n{'='*80}")
            print(f"MODELO: {model_name}")
            print(f"{'='*80}\n")
            
            # Por classe
            for dataset_info in self.test_datasets:
                dataset_name = dataset_info['name']
                class_id = dataset_info['class_id']
                
                if dataset_name not in self.metrics[model_name]:
                    continue
                
                print(f"Classe {class_id}:")
                print("-" * 80)
                
                dataset_metrics = self.metrics[model_name][dataset_name]
                
                # Agregar
                all_wmape = np.concatenate([df['wmape'].values for df in dataset_metrics])
                all_wmape = all_wmape[np.isfinite(all_wmape)]
                
                all_mape_acc = []
                all_mae_acc = []
                all_mape_daily = []
                all_mae_daily = []
                
                for df in dataset_metrics:
                    # Acumulado
                    valid = ~(pd.isna(df['real_acc']) | pd.isna(df['pred_acc'])) & (df['real_acc'] > 0)
                    if valid.any():
                        mape = 100 * np.abs(df.loc[valid, 'real_acc'] - df.loc[valid, 'pred_acc']) / df.loc[valid, 'real_acc']
                        mae_vals = np.abs(df.loc[valid, 'real_acc'] - df.loc[valid, 'pred_acc'])
                        
                        all_mape_acc.extend(mape[np.isfinite(mape)])
                        all_mae_acc.extend(mae_vals[np.isfinite(mae_vals)])
                    
                    # Diário
                    valid_daily = ~(pd.isna(df['real_daily_avg']) | pd.isna(df['pred_daily_avg'])) & (df['real_daily_avg'] > 0)
                    if valid_daily.any():
                        mape_daily = 100 * np.abs(df.loc[valid_daily, 'real_daily_avg'] - df.loc[valid_daily, 'pred_daily_avg']) / df.loc[valid_daily, 'real_daily_avg']
                        mae_daily = np.abs(df.loc[valid_daily, 'real_daily_avg'] - df.loc[valid_daily, 'pred_daily_avg'])
                        
                        all_mape_daily.extend(mape_daily[np.isfinite(mape_daily)])
                        all_mae_daily.extend(mae_daily[np.isfinite(mae_daily)])
                
                n_series = sum(len(df) for df in dataset_metrics)
                print(f"  Séries: {n_series}")
                print()
                
                if len(all_wmape) > 0:
                    wmape_stats = MetricStats.from_array(all_wmape)
                    print(f"  WMAPE (normalizado):")
                    print(f"    Mediana: {wmape_stats.median:.4f} | P90: {wmape_stats.p90:.4f} | P95: {wmape_stats.p95:.4f}")
                
                if all_mape_acc:
                    mape_acc_stats = MetricStats.from_array(np.array(all_mape_acc))
                    print(f"  MAPE Acumulado:")
                    print(f"    Mediana: {mape_acc_stats.median:.2f}% | P90: {mape_acc_stats.p90:.2f}% | P95: {mape_acc_stats.p95:.2f}%")
                
                if all_mae_acc:
                    print(f"  MAE Acumulado:")
                    print(f"    Mediana: {np.median(all_mae_acc):.2f} | P90: {np.percentile(all_mae_acc, 90):.2f}")
                
                if all_mape_daily:
                    mape_daily_stats = MetricStats.from_array(np.array(all_mape_daily))
                    print(f"  MAPE Diário:")
                    print(f"    Mediana: {mape_daily_stats.median:.2f}% | P90: {mape_daily_stats.p90:.2f}% | P95: {mape_daily_stats.p95:.2f}%")
                
                if all_mae_daily:
                    print(f"  MAE Diário:")
                    print(f"    Mediana: {np.median(all_mae_daily):.2f} | P90: {np.percentile(all_mae_daily, 90):.2f}")
                
                print()
            
            # Agregado global
            print(f"AGREGADO (todas as classes):")
            print("-" * 80)
            
            all_wmape_global = []
            all_mape_acc_global = []
            all_mape_daily_global = []
            
            for dataset_name, dataset_metrics in self.metrics[model_name].items():
                wmape_vals = np.concatenate([df['wmape'].values for df in dataset_metrics])
                all_wmape_global.extend(wmape_vals[np.isfinite(wmape_vals)])
                
                for df in dataset_metrics:
                    valid = ~(pd.isna(df['real_acc']) | pd.isna(df['pred_acc'])) & (df['real_acc'] > 0)
                    if valid.any():
                        mape = 100 * np.abs(df.loc[valid, 'real_acc'] - df.loc[valid, 'pred_acc']) / df.loc[valid, 'real_acc']
                        all_mape_acc_global.extend(mape[np.isfinite(mape)])
                    
                    valid_daily = ~(pd.isna(df['real_daily_avg']) | pd.isna(df['pred_daily_avg'])) & (df['real_daily_avg'] > 0)
                    if valid_daily.any():
                        mape_daily = 100 * np.abs(df.loc[valid_daily, 'real_daily_avg'] - df.loc[valid_daily, 'pred_daily_avg']) / df.loc[valid_daily, 'real_daily_avg']
                        all_mape_daily_global.extend(mape_daily[np.isfinite(mape_daily)])
            
            if all_wmape_global:
                wmape_global = MetricStats.from_array(np.array(all_wmape_global))
                print(f"  WMAPE: Mediana={wmape_global.median:.4f}, P90={wmape_global.p90:.4f}")
            
            if all_mape_acc_global:
                mape_acc_global = MetricStats.from_array(np.array(all_mape_acc_global))
                print(f"  MAPE Acumulado: Mediana={mape_acc_global.median:.2f}%, P90={mape_acc_global.p90:.2f}%")
            
            if all_mape_daily_global:
                mape_daily_global = MetricStats.from_array(np.array(all_mape_daily_global))
                print(f"  MAPE Diário: Mediana={mape_daily_global.median:.2f}%, P90={mape_daily_global.p90:.2f}%")
            
            print()
        
        print("=" * 80)
        print()
    
    def compare_classes(
        self,
        model_name: str,
        metric: str = 'mape_acc',
        figsize: tuple = (14, 6)
    ):
        """Compara performance entre classes"""
        if model_name not in self.metrics:
            raise ValueError(f"Modelo {model_name} não encontrado")
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        class_data = {}
        
        for dataset_name, dataset_metrics in self.metrics[model_name].items():
            class_id = self.predictions[model_name][dataset_name]['class_id']
            
            if metric == 'mape_acc':
                values = []
                for df in dataset_metrics:
                    valid = ~(pd.isna(df['real_acc']) | pd.isna(df['pred_acc'])) & (df['real_acc'] > 0)
                    if valid.any():
                        mape = 100 * np.abs(df.loc[valid, 'real_acc'] - df.loc[valid, 'pred_acc']) / df.loc[valid, 'real_acc']
                        values.extend(mape[np.isfinite(mape)])
            
            elif metric == 'mape_daily':
                values = []
                for df in dataset_metrics:
                    valid = ~(pd.isna(df['real_daily_avg']) | pd.isna(df['pred_daily_avg'])) & (df['real_daily_avg'] > 0)
                    if valid.any():
                        mape = 100 * np.abs(df.loc[valid, 'real_daily_avg'] - df.loc[valid, 'pred_daily_avg']) / df.loc[valid, 'real_daily_avg']
                        values.extend(mape[np.isfinite(mape)])
            
            elif metric == 'mae_acc':
                values = []
                for df in dataset_metrics:
                    valid = ~(pd.isna(df['real_acc']) | pd.isna(df['pred_acc']))
                    if valid.any():
                        mae_vals = np.abs(df.loc[valid, 'real_acc'] - df.loc[valid, 'pred_acc'])
                        values.extend(mae_vals[np.isfinite(mae_vals)])
            
            elif metric == 'mae_daily':
                values = []
                for df in dataset_metrics:
                    valid = ~(pd.isna(df['real_daily_avg']) | pd.isna(df['pred_daily_avg']))
                    if valid.any():
                        mae_vals = np.abs(df.loc[valid, 'real_daily_avg'] - df.loc[valid, 'pred_daily_avg'])
                        values.extend(mae_vals[np.isfinite(mae_vals)])
            
            else:
                values = np.concatenate([df[metric].values for df in dataset_metrics])
                values = values[np.isfinite(values)]
            
            if len(values) > 0:
                class_data[f"Classe {class_id}"] = np.array(values)
        
        if not class_data:
            print("Nenhum dado válido")
            return
        
        # Boxplot
        ax1 = axes[0]
        bp = ax1.boxplot(list(class_data.values()), tick_labels=list(class_data.keys()), patch_artist=True, showfliers=False)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(class_data)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_ylabel(metric.upper(), fontsize=12, fontweight='bold')
        ax1.set_title(f'Distribuição por Classe', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Ranking
        ax2 = axes[1]
        class_names = list(class_data.keys())
        medians = [np.median(v) for v in class_data.values()]
        
        sorted_idx = np.argsort(medians)
        class_names_sorted = [class_names[i] for i in sorted_idx]
        medians_sorted = [medians[i] for i in sorted_idx]
        colors_sorted = [colors[i] for i in sorted_idx]
        
        bars = ax2.barh(range(len(class_names_sorted)), medians_sorted, color=colors_sorted, alpha=0.8)
        
        ax2.set_yticks(range(len(class_names_sorted)))
        ax2.set_yticklabels(class_names_sorted)
        ax2.set_xlabel(f'{metric.upper()} (Mediana)', fontsize=12, fontweight='bold')
        ax2.set_title('Ranking por Classe', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        for i, (bar, val) in enumerate(zip(bars, medians_sorted)):
            ax2.text(val, i, f' {val:.2f}', va='center', fontsize=10, fontweight='bold')
        
        plt.suptitle(f'{model_name} | {metric.upper()} | Target: {self.target}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def compare_models(
        self,
        class_id: Optional[int] = None,
        metric: str = 'mape_acc',
        figsize: tuple = (14, 6)
    ):
        """Compara modelos"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        model_data = {}
        
        for model_name in self.model_names:
            if model_name not in self.metrics:
                continue
            
            values = []
            
            for dataset_info in self.test_datasets:
                dataset_name = dataset_info['name']
                ds_class_id = dataset_info['class_id']
                
                if class_id is not None and ds_class_id != class_id:
                    continue
                
                if dataset_name not in self.metrics[model_name]:
                    continue
                
                dataset_metrics = self.metrics[model_name][dataset_name]
                
                if metric in ['mape_acc', 'mae_acc', 'mape_daily', 'mae_daily']:
                    for df in dataset_metrics:
                        col_real = 'real_acc' if 'acc' in metric else 'real_daily_avg'
                        col_pred = 'pred_acc' if 'acc' in metric else 'pred_daily_avg'
                        
                        valid = ~(pd.isna(df[col_real]) | pd.isna(df[col_pred])) & (df[col_real] > 0)
                        
                        if valid.any():
                            if 'mape' in metric:
                                vals = 100 * np.abs(df.loc[valid, col_real] - df.loc[valid, col_pred]) / df.loc[valid, col_real]
                            else:
                                vals = np.abs(df.loc[valid, col_real] - df.loc[valid, col_pred])
                            
                            values.extend(vals[np.isfinite(vals)])
                else:
                    vals = np.concatenate([df[metric].values for df in dataset_metrics])
                    values.extend(vals[np.isfinite(vals)])
            
            if values:
                model_data[model_name] = np.array(values)
        
        if not model_data:
            print("Nenhum dado válido")
            return
        
        # Boxplot
        ax1 = axes[0]
        bp = ax1.boxplot(list(model_data.values()), tick_labels=list(model_data.keys()), patch_artist=True, showfliers=False)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_data)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        
        ax1.set_ylabel(metric.upper(), fontsize=12, fontweight='bold')
        ax1.set_title('Distribuição', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Ranking
        ax2 = axes[1]
        model_names = list(model_data.keys())
        medians = [np.median(v) for v in model_data.values()]
        
        sorted_idx = np.argsort(medians)
        model_names_sorted = [model_names[i] for i in sorted_idx]
        medians_sorted = [medians[i] for i in sorted_idx]
        colors_sorted = [colors[i] for i in sorted_idx]
        
        bars = ax2.barh(range(len(model_names_sorted)), medians_sorted, color=colors_sorted, alpha=0.8)
        
        ax2.set_yticks(range(len(model_names_sorted)))
        ax2.set_yticklabels(model_names_sorted)
        ax2.set_xlabel(f'{metric.upper()} (Mediana)', fontsize=12, fontweight='bold')
        ax2.set_title('Ranking', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        for i, (bar, val) in enumerate(zip(bars, medians_sorted)):
            ax2.text(val, i, f' {val:.2f}', va='center', fontsize=10, fontweight='bold')
        
        class_str = f"Classe {class_id}" if class_id is not None else "Todas as classes"
        plt.suptitle(f'{class_str} | {metric.upper()} | Target: {self.target}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_prediction(
        self,
        model_name: str,
        class_id: int,
        veiculo_id: int,
        window: int = 0,
        figsize: tuple = (12, 6)
    ):
        """
        Plota predição com intervalos de confiança para um veículo específico
        
        Parameters:
        -----------
        model_name : str
            Nome do modelo
        class_id : int
            ID da classe
        veiculo_id : int
            ID do veículo
        window : int, optional
            Índice da janela (padrão: 0)
        figsize : tuple, optional
            Tamanho da figura (padrão: (12, 6))
        
        Examples:
        ---------
        >>> evaluator.plot_prediction(
        ...     model_name='DLinearModel',
        ...     class_id=0,
        ...     veiculo_id=123,
        ...     window=0
        ... )
        """
        # Validação: Encontrar dataset da classe
        dataset_name = None
        for ds_info in self.test_datasets:
            if ds_info['class_id'] == class_id:
                dataset_name = ds_info['name']
                break
        
        if dataset_name is None:
            raise ValueError(
                f"Classe {class_id} não encontrada. "
                f"Classes disponíveis: {[ds['class_id'] for ds in self.test_datasets]}"
            )
        
        # Validação: Verificar se modelo tem predições
        if model_name not in self.predictions:
            raise ValueError(
                f"Modelo {model_name} não tem predições. "
                f"Execute predict_all() primeiro."
            )
        
        if dataset_name not in self.predictions[model_name]:
            raise ValueError(
                f"Dataset {dataset_name} não encontrado para {model_name}. "
                f"Datasets disponíveis: {list(self.predictions[model_name].keys())}"
            )
        
        # Validação: Verificar janela
        windows = self.test_windows[dataset_name]
        
        if window < 0 or window >= len(windows):
            raise ValueError(
                f"Janela {window} inválida. "
                f"Janelas disponíveis: 0-{len(windows)-1}"
            )
        
        test_window = windows[window]
        
        # Validação: Verificar veículo
        if veiculo_id not in test_window.ids:
            raise ValueError(
                f"Veículo {veiculo_id} não encontrado na janela {window}. "
                f"IDs disponíveis: {test_window.ids[:10]}"
            )
        
        # Obter índice e séries
        idx = test_window.ids.index(veiculo_id)
        real_ts_scaled = test_window.output_series[idx]
        
        pred_data = self.predictions[model_name][dataset_name]
        pred_ts = pred_data['predictions'][window][idx]
        quantiles = pred_data['quantiles']
        
        history_end_date = test_window.input_series[idx].end_time()
        
        # Desnormalizar real
        real_values = self.vp.inverse_transform(
            vehicle_id=veiculo_id,
            scaled_values=real_ts_scaled.values().flatten(),
            date=history_end_date
        )
        
        # Criar figura
        time_idx = real_ts_scaled.time_index
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plotar valores reais
        ax.plot(
            time_idx, 
            real_values, 
            label='Real', 
            marker='o', 
            linewidth=2, 
            markersize=6, 
            color='#2E86AB',
            zorder=3
        )
        
        # Verificar estrutura da predição
        pred_shape = pred_ts.all_values(copy=False).shape
        # Shape esperado: (n_timesteps, n_quantiles, n_samples)
        # Exemplo: (7, 7, 1)
        
        if quantiles and len(pred_shape) == 3 and pred_shape[1] == len(quantiles):
            # Estrutura: (timesteps, quantiles, samples)
            # Cada "componente" (eixo 1) é um quantil
            
            # Extrair mediana
            median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
            
            pred_median_scaled = pred_ts.all_values(copy=False)[:, median_idx, 0]
            pred_median = self.vp.inverse_transform(
                vehicle_id=veiculo_id,
                scaled_values=pred_median_scaled,
                date=history_end_date
            )
            
            ax.plot(
                time_idx, 
                pred_median, 
                label='Predito (Mediana)', 
                marker='s', 
                linewidth=2, 
                markersize=6, 
                color='#F18F01',
                zorder=3
            )
            
            # Intervalo de confiança 90% (q05 - q95)
            if 0.05 in quantiles and 0.95 in quantiles:
                q05_idx = quantiles.index(0.05)
                q95_idx = quantiles.index(0.95)
                
                pred_q05_scaled = pred_ts.all_values(copy=False)[:, q05_idx, 0]
                pred_q95_scaled = pred_ts.all_values(copy=False)[:, q95_idx, 0]
                
                pred_q05 = self.vp.inverse_transform(
                    vehicle_id=veiculo_id,
                    scaled_values=pred_q05_scaled,
                    date=history_end_date
                )
                pred_q95 = self.vp.inverse_transform(
                    vehicle_id=veiculo_id,
                    scaled_values=pred_q95_scaled,
                    date=history_end_date
                )
                
                ax.fill_between(
                    time_idx, 
                    pred_q05, 
                    pred_q95, 
                    alpha=0.2, 
                    color='#F18F01', 
                    label='IC 90% (q05-q95)',
                    zorder=1
                )
            
            # Intervalo de confiança 50% (q25 - q75)
            if 0.25 in quantiles and 0.75 in quantiles:
                q25_idx = quantiles.index(0.25)
                q75_idx = quantiles.index(0.75)
                
                pred_q25_scaled = pred_ts.all_values(copy=False)[:, q25_idx, 0]
                pred_q75_scaled = pred_ts.all_values(copy=False)[:, q75_idx, 0]
                
                pred_q25 = self.vp.inverse_transform(
                    vehicle_id=veiculo_id,
                    scaled_values=pred_q25_scaled,
                    date=history_end_date
                )
                pred_q75 = self.vp.inverse_transform(
                    vehicle_id=veiculo_id,
                    scaled_values=pred_q75_scaled,
                    date=history_end_date
                )
                
                ax.fill_between(
                    time_idx, 
                    pred_q25, 
                    pred_q75, 
                    alpha=0.3, 
                    color='#F18F01', 
                    label='IC 50% (q25-q75)',
                    zorder=2
                )
        
        else:
            # Predição pontual (sem quantis)
            pred_values = self.vp.inverse_transform(
                vehicle_id=veiculo_id,
                scaled_values=pred_ts.values().flatten(),
                date=history_end_date
            )
            
            ax.plot(
                time_idx, 
                pred_values, 
                label='Predito', 
                marker='s', 
                linewidth=2, 
                markersize=6, 
                color='#F18F01',
                zorder=3
            )
        
        # Obter métricas do veículo
        df_metrics = self.metrics[model_name][dataset_name][window]
        vehicle_metrics = df_metrics[df_metrics['veiculo_id'] == veiculo_id]
        
        if len(vehicle_metrics) == 0:
            raise ValueError(f"Métricas não encontradas para veículo {veiculo_id}")
        
        vm = vehicle_metrics.iloc[0]
        
        # Calcular MAPE acumulado
        mape_acc = (
            100 * np.abs(vm['real_acc'] - vm['pred_acc']) / vm['real_acc'] 
            if vm['real_acc'] > 0 and not np.isnan(vm['real_acc']) 
            else np.nan
        )
        
        # Calcular MAPE diário
        mape_daily = (
            100 * np.abs(vm['real_daily_avg'] - vm['pred_daily_avg']) / vm['real_daily_avg'] 
            if vm['real_daily_avg'] > 0 and not np.isnan(vm['real_daily_avg']) 
            else np.nan
        )
        
        # Configurar eixos
        unit = self.target.split('_')[0]
        
        ax.set_xlabel('Data', fontsize=12, fontweight='bold')
        ax.set_ylabel(unit, fontsize=12, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Título com métricas
        title_line1 = f"Veículo {veiculo_id} | Classe {class_id} | Janela {window} | {model_name}"
        
        title_line2 = (
            f"WMAPE={vm['wmape']:.2f} | "
            f"R²={vm['r2']:.3f}"
        )
        
        title_line3 = (
            f"ACUMULADO: Real={vm['real_acc']:.1f}, Pred={vm['pred_acc']:.1f}, MAPE={mape_acc:.1f}%"
        )
        
        title_line4 = (
            f"DIÁRIO: Real={vm['real_daily_avg']:.2f}, Pred={vm['pred_daily_avg']:.2f}, MAPE={mape_daily:.1f}%"
        )
        
        title = f"{title_line1}\n{title_line2}\n{title_line3} | {title_line4}"
        
        ax.set_title(title, fontsize=10, fontweight='bold')
        
        # Rotacionar labels do eixo x
        plt.xticks(rotation=45, ha='right')
        
        # Ajustar layout
        plt.tight_layout()
        
        # Mostrar
        plt.show()    
    def export_results(self, output_dir: Optional[str] = None) -> Dict[str, Dict[str, Path]]:
        """Exporta resultados por classe"""
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = Path('.')
        
        exported = {}
        
        print("💾 Exportando resultados...")
        
        for model_name in self.metrics.keys():
            exported[model_name] = {}
            
            for dataset_name, dataset_metrics in self.metrics[model_name].items():
                class_id = self.predictions[model_name][dataset_name]['class_id']
                
                all_results = []
                for df in dataset_metrics:
                    df_copy = df.copy()
                    df_copy['model'] = model_name
                    
                    # MAPE acumulado
                    valid = ~(pd.isna(df_copy['real_acc']) | pd.isna(df_copy['pred_acc'])) & (df_copy['real_acc'] > 0)
                    df_copy['mape_acc'] = np.nan
                    if valid.any():
                        df_copy.loc[valid, 'mape_acc'] = 100 * np.abs(
                            df_copy.loc[valid, 'real_acc'] - df_copy.loc[valid, 'pred_acc']
                        ) / df_copy.loc[valid, 'real_acc']
                    
                    # MAPE diário
                    valid_daily = ~(pd.isna(df_copy['real_daily_avg']) | pd.isna(df_copy['pred_daily_avg'])) & (df_copy['real_daily_avg'] > 0)
                    df_copy['mape_daily'] = np.nan
                    if valid_daily.any():
                        df_copy.loc[valid_daily, 'mape_daily'] = 100 * np.abs(
                            df_copy.loc[valid_daily, 'real_daily_avg'] - df_copy.loc[valid_daily, 'pred_daily_avg']
                        ) / df_copy.loc[valid_daily, 'real_daily_avg']
                    
                    all_results.append(df_copy)
                
                df_consolidated = pd.concat(all_results, ignore_index=True)
                
                filename = f'test_{model_name}_class{class_id}_{self.target}.csv'
                filepath = output_path / filename
                
                df_consolidated.to_csv(filepath, index=False)
                exported[model_name][dataset_name] = filepath
                
                if self.verbose:
                    file_size = filepath.stat().st_size / 1024
                    print(f"  ✓ {model_name} Classe {class_id}: {file_size:.1f} KB")
        
        if self.verbose:
            print()
        
        return exported
    
    def clear_cache(self):
        """Remove cache de predições"""
        cache_files = list(self.cache_dir.glob('pred_*.pkl'))
        
        if not cache_files:
            print("Nenhum arquivo de cache encontrado")
            return
        
        print(f"🗑️  Removendo {len(cache_files)} arquivo(s) de cache...")
        
        for cache_file in cache_files:
            try:
                cache_file.unlink()
                if self.verbose:
                    print(f"  ✓ {cache_file.name}")
            except Exception as e:
                warnings.warn(f"Erro ao remover {cache_file.name}: {e}")
        
        print("✓ Cache limpo")
    
    def __repr__(self) -> str:
        return (
            f"TestEvaluator(\n"
            f"  target='{self.target}',\n"
            f"  n_datasets={len(self.test_datasets)},\n"
            f"  n_models={len(self.model_names)},\n"
            f"  cache_dir='{self.cache_dir}',\n"
            f"  likelihood={self.likelihood}\n"
            f")"
        )
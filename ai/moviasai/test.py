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
                exported[f"{dataset_name}_class_id{class_id}_hist{history_size}"] = filepath
                
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

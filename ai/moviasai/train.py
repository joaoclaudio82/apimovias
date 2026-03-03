import os
import numpy as np
import datetime
import warnings
import json
from pathlib import Path

from typing import List, Optional, Callable, Dict, Any, Tuple, Type, Union

from dataclasses import dataclass

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from darts import TimeSeries
from darts.metrics import wmape
from darts.utils.callbacks import TFMProgressBar
from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel

from moviasai.dataset_loader import SlicedDataset, Window


@dataclass
class MetricStats:
    """Estatísticas agregadas de métricas com percentis expandidos"""
    n_valid: int
    mean: float
    median: float
    std: float
    min: float
    max: float
    p10: float
    p25: float
    p75: float
    p90: float
    p95: float
    p99: float
    
    @classmethod
    def from_array(cls, values: np.ndarray) -> 'MetricStats':
        """
        Cria estatísticas a partir de um array
        
        Parameters:
        -----------
        values : np.ndarray
            Array com valores da métrica
        
        Returns:
        --------
        MetricStats
            Estatísticas calculadas
        """
        valid_values = values[np.isfinite(values)]
        
        if len(valid_values) == 0:
            return cls(
                n_valid=0,
                mean=np.nan,
                median=np.nan,
                std=np.nan,
                min=np.nan,
                max=np.nan,
                p10=np.nan,
                p25=np.nan,
                p75=np.nan,
                p90=np.nan,
                p95=np.nan,
                p99=np.nan
            )
        
        return cls(
            n_valid=len(valid_values),
            mean=float(np.mean(valid_values)),
            median=float(np.median(valid_values)),
            std=float(np.std(valid_values)),
            min=float(np.min(valid_values)),
            max=float(np.max(valid_values)),
            p10=float(np.percentile(valid_values, 10)),
            p25=float(np.percentile(valid_values, 25)),
            p75=float(np.percentile(valid_values, 75)),
            p90=float(np.percentile(valid_values, 90)),
            p95=float(np.percentile(valid_values, 95)),
            p99=float(np.percentile(valid_values, 99))
        )
    
    def __str__(self) -> str:
        """Representação em string formatada"""
        if self.n_valid == 0:
            return "Nenhum valor válido"
        
        return (
            f"N válidos: {self.n_valid}\n"
            f"  • Média: {self.mean:.4f}\n"
            f"  • Mediana: {self.median:.4f}\n"
            f"  • Std: {self.std:.4f}\n"
            f"  • Min: {self.min:.4f}\n"
            f"  • P10: {self.p10:.4f}\n"
            f"  • P25: {self.p25:.4f}\n"
            f"  • P75: {self.p75:.4f}\n"
            f"  • P90: {self.p90:.4f}\n"
            f"  • P95: {self.p95:.4f}\n"
            f"  • P99: {self.p99:.4f}\n"
            f"  • Max: {self.max:.4f}"
        )


class TrainerOptionsBuilder:
    """Construtor de opções para PyTorch Lightning Trainer"""
    
    def __init__(
        self,
        early_stopper_args: Optional[Dict[str, Any]] = None,
        add_progress_bar: bool = True,
        **kwargs
    ):
        self.early_stopper_args = early_stopper_args or {}
        self.add_progress_bar = add_progress_bar
        self.kwargs = kwargs
    
    def build(self) -> Dict[str, Any]:
        """Constrói dicionário de opções do trainer"""
        callbacks = []
        
        if self.early_stopper_args:
            callbacks.append(EarlyStopping(**self.early_stopper_args))
        
        if self.add_progress_bar:
            progress_bar = TFMProgressBar(enable_train_bar_only=True)
            callbacks.append(progress_bar)
        
        return {
            "callbacks": callbacks,
            **self.kwargs
        }


class ModelBuilder:
    """Construtor de modelos Darts com suporte a checkpoints"""
    
    def __init__(
        self,
        model_cls: Type,
        options_builder: Optional[TrainerOptionsBuilder] = None,
        **kwargs
    ):
        self.model_cls = model_cls
        self.options_builder = options_builder
        self.kwargs = kwargs
        
        # Detectar tipo de modelo
        self.is_torch_model = issubclass(model_cls, TorchForecastingModel)
        self.is_global_model = issubclass(model_cls, GlobalForecastingModel)
    
    @property
    def name(self) -> str:
        """Nome do modelo"""
        return self.model_cls.__name__
    
    @staticmethod
    def _get_model_name(idx_train: int) -> str:
        """Gera nome do modelo"""
        return f'model_{idx_train:02d}'
    
    def load_from_checkpoint(
        self,
        work_dir: Path,
        idx_train: int
    ) -> Optional[Any]:
        """Carrega modelo de checkpoint"""
        try:
            if self.is_torch_model:
                model = self.model_cls.load_from_checkpoint(
                    model_name=self._get_model_name(idx_train),
                    work_dir=str(work_dir),
                    best=True
                )
            else:
                model_path = work_dir / f'{self._get_model_name(idx_train)}.pkl'
                if model_path.exists():
                    model = self.model_cls.load(str(model_path))
                else:
                    return None
            
            return model
        
        except Exception as e:
            warnings.warn(f"Não foi possível carregar checkpoint: {e}")
            return None
    
    def build(
        self,
        work_dir: Path,
        idx_train: int,
        incremental: bool = False,
        force_reset: bool = False,
    ) -> Any:
        """
        Cria instância do modelo com configurações
        """
        model_kwargs = {**self.kwargs}
        
        if self.is_torch_model:
            model_kwargs.update({
                "work_dir": str(work_dir),
                "model_name": self._get_model_name(idx_train),
                "force_reset": force_reset
            })
            
            # Configurações do trainer
            if self.options_builder:
                pl_trainer_kwargs = self.options_builder.build()
            else:
                # Criar kwargs padrão se não fornecido
                pl_trainer_kwargs = {
                    "callbacks": []
                }
            
            # SEMPRE adicionar logger (obrigatório)
            pl_trainer_kwargs['logger'] = TensorBoardLogger(
                save_dir=work_dir / "tb_logs",
                name=f"window_{idx_train:02d}"
            )
            
            model_kwargs['pl_trainer_kwargs'] = pl_trainer_kwargs
        
        # Criar modelo
        model = self.model_cls(**model_kwargs)
        
        # Carregar pesos anteriores se incremental
        if self.is_torch_model and incremental and idx_train > 0:
            self._load_previous_weights(model, work_dir, idx_train)
        
        return model
    
    def _load_previous_weights(
        self,
        model: Any,
        work_dir: Path,
        idx_train: int
    ) -> None:
        """Carrega pesos do modelo anterior"""
        previous_model_name = self._get_model_name(idx_train - 1)
        print(f"🔄 Carregando pesos do modelo anterior: {previous_model_name}")
        
        try:
            model.load_weights_from_checkpoint(
                model_name=previous_model_name,
                work_dir=str(work_dir),
                best=True
            )
            print("✓ Pesos carregados com sucesso")
        except Exception as e:
            warnings.warn(f"Não foi possível carregar pesos: {e}")
    
    def save_model(
        self,
        model: Any,
        work_dir: Path,
        idx_train: int
    ) -> None:
        """Salva modelo em disco"""
        model_name = self._get_model_name(idx_train)
        
        if self.is_torch_model:
            print(f"✓ Checkpoint PyTorch: {model_name}")
        else:
            model_path = work_dir / f'{model_name}.pkl'
            print(f"💾 Salvando modelo: {model_path.name}")
            model.save(str(model_path))
            print("✓ Modelo salvo com sucesso")


class TrainPipeline:
    """
    Pipeline para treinamento e avaliação de modelos de séries temporais
    """
    
    def __init__(
        self,
        train_data: Union[List[Window], List[SlicedDataset], SlicedDataset],
        val_data: Union[List[Window], List[SlicedDataset], SlicedDataset],
        history_size: int = 28,
        forecast_horizon: int = 7,
        experiments_dir: Optional[str] = None,
        experiment_label: Optional[str] = None,
        eval_metric_fn: Callable = wmape,
    ):
        # Normalizar entrada
        self.train_data = self._normalize_input(train_data)
        self.val_data = self._normalize_input(val_data)
        
        # Validações
        self._validate_inputs(
            self.train_data,
            self.val_data,
            history_size,
            forecast_horizon
        )
        
        self.history_size = history_size
        self.forecast_horizon = forecast_horizon
        self.eval_metric_fn = eval_metric_fn
        
        # Achatar validação para fit
        self._val_data = self._flatten_windows(self.val_data)
        
        # Configurar diretórios
        self.experiments_dir = self._setup_experiments_dir(experiments_dir)
        self.experiment_label = experiment_label or self._generate_experiment_label()
    
    @staticmethod
    def _normalize_input(
        data: Union[List[Window], List[SlicedDataset], SlicedDataset, Window]
    ) -> List[Union[Window, SlicedDataset]]:
        """Normaliza entrada para lista"""
        if isinstance(data, list):
            return data
        return [data]
    
    @staticmethod
    def _validate_inputs(
        train_data: List[Union[Window, SlicedDataset]],
        val_data: List[Union[Window, SlicedDataset]],
        history_size: int,
        forecast_horizon: int
    ) -> None:
        """Valida parâmetros"""
        if not train_data:
            raise ValueError("train_data está vazio")
        if not val_data:
            raise ValueError("val_data está vazio")
        if history_size <= 0:
            raise ValueError(f"history_size deve ser > 0, recebido: {history_size}")
        if forecast_horizon <= 0:
            raise ValueError(f"forecast_horizon deve ser > 0, recebido: {forecast_horizon}")
    
    @staticmethod
    def _flatten_windows(
        windows: List[Union[Window, SlicedDataset]]
    ) -> Dict[str, List[TimeSeries]]:
        """Achata janelas em listas únicas"""
        if not windows:
            raise ValueError("Lista de windows está vazia")
        
        all_series = []
        all_past_covariates = []
        all_future_covariates = []
        
        has_past_covariates = any(w.past_covariates is not None for w in windows)
        has_future_covariates = any(w.future_covariates is not None for w in windows)
        
        for window in windows:
            all_series.extend(window.series)
            
            if window.past_covariates is not None:
                all_past_covariates.extend(window.past_covariates)
            elif has_past_covariates:
                all_past_covariates.extend([None] * len(window))
            
            if window.future_covariates is not None:
                all_future_covariates.extend(window.future_covariates)
            elif has_future_covariates:
                all_future_covariates.extend([None] * len(window))
        
        return {
            'series': all_series,
            'past_covariates': all_past_covariates if all_past_covariates else None,
            'future_covariates': all_future_covariates if all_future_covariates else None
        }
    
    @staticmethod
    def _setup_experiments_dir(experiments_dir: Optional[str]) -> Path:
        """Configura diretório de experimentos"""
        if experiments_dir is None:
            experiments_dir = os.getcwd()
        path = Path(experiments_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def _generate_experiment_label() -> str:
        """Gera label do experimento"""
        now = datetime.datetime.now()
        return now.strftime("%Y-%m-%d_%H-%M-%S")
    
    def get_train_work_dir(self, model_name: str) -> Path:
        """Cria diretório de trabalho"""
        model_path = self.experiments_dir / model_name / self.experiment_label
        model_path.mkdir(parents=True, exist_ok=True)
        return model_path
    
    def _compute_predictions(
        self,
        model: Any,
        data: Union[Window, SlicedDataset]
    ) -> List:
        """Computa predições"""
        return model.predict(
            n=self.forecast_horizon,
            series=data.input_series,
            past_covariates=data.input_past_covariates if model.supports_past_covariates else None,
            future_covariates=data.output_future_covariates if model.supports_future_covariates else None
        )
    
    def _eval_metrics(
        self,
        model: Any,
        train_data: Union[Window, SlicedDataset],
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Avalia métricas de treino e validação
        
        Returns:
        --------
        tuple
            (train_metrics, val_window_metrics)
            - train_metrics: array com métrica de cada série de treino
            - val_window_metrics: lista de arrays, um por janela de validação
        """
        # Predições de treino
        pred_train = self._compute_predictions(model, train_data)
        
        # Métricas de treino (valores individuais)
        train_metrics = np.array(
            self.eval_metric_fn(train_data.output_series, pred_train),
            dtype=float
        )
        train_metrics = train_metrics[np.isfinite(train_metrics)]
        
        # Métricas para cada janela de validação (valores individuais)
        val_window_metrics = []
        
        for val_data in self.val_data:
            pred_val = self._compute_predictions(model, val_data)
            
            val_metrics = np.array(
                self.eval_metric_fn(val_data.output_series, pred_val),
                dtype=float
            )
            val_metrics = val_metrics[np.isfinite(val_metrics)]
            
            val_window_metrics.append(val_metrics)
        
        return train_metrics, val_window_metrics
    
    def _log_metrics_to_tensorboard(
        self,
        model: Any,
        idx_train: int,
        train_stats: MetricStats,
        val_stats_list: List[MetricStats],
        is_torch_model: bool
    ) -> None:
        """Registra métricas no TensorBoard (apenas PyTorch)"""
        if not is_torch_model:
            return
        
        metric_name = self.eval_metric_fn.__name__.lower()
        logger = model.trainer.logger.experiment
        
        # Métricas de treino
        for stat_name in ['median', 'mean', 'std', 'min', 'max', 'p10', 'p25', 'p75', 'p90', 'p95', 'p99']:
            logger.add_scalar(
                f"{metric_name}_train_{stat_name}",
                getattr(train_stats, stat_name),
                global_step=idx_train
            )
        
        # Métricas de validação por janela
        for idx_val, val_stats in enumerate(val_stats_list):
            for stat_name in ['median', 'mean', 'std', 'min', 'max', 'p90', 'p95', 'p99']:
                logger.add_scalar(
                    f"{metric_name}_val_window{idx_val}_{stat_name}",
                    getattr(val_stats, stat_name),
                    global_step=idx_train
                )

    def _save_model_standard_format(
        self,
        model: Any,
        work_dir: Path,
        idx_train: int,
        model_type: str,
        is_torch_model: bool
    ) -> None:
        """
        Salva modelo no formato padrão .pkl com metadados
        """
        try:
            model_name = f'model_{idx_train:02d}'
            target_pkl = work_dir / f"{model_name}.pkl"
            
            # Se já existe, pular
            if target_pkl.exists() and target_pkl.is_file():
                return
            
            print(f"💾 Salvando formato padrão: {target_pkl.name}")
            model.save(str(target_pkl))
            
            # Criar metadados
            parts = work_dir.name.split('_')
            metadata = {
                'dataset': work_dir.name,
                'created_at': datetime.datetime.now().isoformat(),
                'model_type': model_type,
                'is_torch_model': is_torch_model,
                'format': 'pickle_file'
            }
            
            # Extrair info do path
            for part in parts:
                if part.startswith('hist'):
                    metadata['input_chunk_length'] = int(part.replace('hist', ''))
                elif part.startswith('fh'):
                    metadata['output_chunk_length'] = int(part.replace('fh', ''))
                elif part.startswith('class'):
                    metadata['class'] = int(part.replace('class', ''))
            
            # Salvar metadados
            metadata_path = work_dir / f"{model_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✓ Formato padrão salvo com metadados")
            
        except Exception as e:
            warnings.warn(f"Não foi possível salvar formato padrão: {e}")
    
    def _save_model_onnx(
        self,
        model: Any,
        work_dir: Path,
        idx_train: int,
        is_torch_model: bool
    ) -> None:
        """Salva modelo no formato ONNX (apenas PyTorch)"""
        if not is_torch_model:
            return
        
        try:
            onnx_path = work_dir / f'model_{idx_train:02d}.onnx'
            print(f"💾 Salvando modelo ONNX: {onnx_path.name}")
            model.to_onnx(str(onnx_path))
            print("✓ Modelo ONNX salvo com sucesso")
        except Exception as e:
            warnings.warn(f"Não foi possível exportar para ONNX: {e}")
    
    def _print_training_header(
        self,
        model_name: str,
        work_dir: Path,
        is_torch_model: bool
    ) -> None:
        """Imprime cabeçalho do treinamento"""
        model_type = "PyTorch" if is_torch_model else "Global"
        
        print("=" * 80)
        print(f"🚀 INICIANDO TREINAMENTO - {model_type}")
        print("=" * 80)
        print(f"Modelo: {model_name}")
        print(f"Diretório: {work_dir}")
        print(f"Janelas de treino: {len(self.train_data)}")
        print(f"Janelas de validação: {len(self.val_data)}")
        
        total_val_series = sum(len(w) for w in self.val_data)
        print(f"Total de séries de validação: {total_val_series}")
        print(f"History size: {self.history_size} dias")
        print(f"Forecast horizon: {self.forecast_horizon} dias")
        print(f"Métrica de avaliação: {self.eval_metric_fn.__name__}")
        print("=" * 80)
        print()
    
    def _print_window_header(
        self,
        idx_train: int,
        train_data: Union[Window, SlicedDataset]
    ) -> None:
        """Imprime cabeçalho da janela"""
        if len(self.train_data) > 1:
            print(f"{'=' * 80}")
            print(f"📊 ETAPA DE TREINO {idx_train + 1}/{len(self.train_data)}")
            print(f"{'=' * 80}")
        
        print(f"Séries de treino: {len(train_data)}")
        
        if isinstance(train_data, SlicedDataset):
            ids = train_data.get_unique_ids()
        else:
            ids = train_data.ids
        
        ids_preview = ids[:5]
        ids_preview_str = f"{ids_preview}, ..." if len(ids) > 5 else str(ids_preview)
        print(f"IDs: {ids_preview_str}")
        print()
    
    def _print_metrics(
        self,
        train_stats: MetricStats,
        val_stats_list: List[MetricStats]
    ) -> None:
        """Imprime estatísticas de métricas"""
        metric_name = self.eval_metric_fn.__name__
        
        print(f"Métricas de Treino ({metric_name}):")
        print(f"  {train_stats}")
        print()
        
        print(f"Métricas de Validação por Janela ({metric_name}):")
        for idx_val, val_stats in enumerate(val_stats_list):
            print(f"  Janela {idx_val + 1}:")
            stats_str = str(val_stats).replace('\n', '\n    ')
            print(f"    {stats_str}")
        print()
    
    def _print_summary(
        self,
        work_dir: Path,
        all_train_metrics: np.ndarray,
        all_val_metrics_by_window: List[List[np.ndarray]]
    ) -> None:
        """
        Imprime resumo final do treinamento
        
        CORRIGIDO: Agrega valores individuais, não médias
        """
        print("=" * 80)
        print("✅ TREINAMENTO CONCLUÍDO")
        print("=" * 80)
        print(f"Total de janelas treinadas: {len(self.train_data)}")
        print(f"Diretório: {work_dir}")
        print()
        
        # Métricas de treino
        train_stats = MetricStats.from_array(all_train_metrics)
        metric_name = self.eval_metric_fn.__name__
        
        print(f"Métricas Agregadas de Treino ({metric_name}):")
        print(f"  {train_stats}")
        print()
        
        # Métricas de validação por janela
        print(f"Métricas de Validação por Janela ({metric_name}):")
        
        for idx_val in range(len(self.val_data)):
            # Coletar TODOS os valores individuais desta janela
            window_metrics = []
            
            for train_idx in range(len(all_val_metrics_by_window)):
                if idx_val < len(all_val_metrics_by_window[train_idx]):
                    window_metrics.append(all_val_metrics_by_window[train_idx][idx_val])
            
            if window_metrics:
                # Concatenar valores individuais
                concatenated = np.concatenate(window_metrics)
                window_stats = MetricStats.from_array(concatenated)
                
                print(f"  Janela {idx_val + 1}:")
                stats_str = str(window_stats).replace('\n', '\n    ')
                print(f"    {stats_str}")
        
        print()
        
        # Métricas totais de validação (CORRIGIDO)
        all_val_metrics = []
        for val_metrics_list in all_val_metrics_by_window:
            for val_metrics in val_metrics_list:
                all_val_metrics.append(val_metrics)
        
        if all_val_metrics:
            concatenated_val = np.concatenate(all_val_metrics)
            val_stats = MetricStats.from_array(concatenated_val)
            
            print(f"Métricas Agregadas Totais de Validação ({metric_name}):")
            print(f"  {val_stats}")
            print()
            
            # ANÁLISE DE OUTLIERS
            print("📊 ANÁLISE DE OUTLIERS:")
            
            # IQR method
            iqr = val_stats.p75 - val_stats.p25
            outlier_threshold = val_stats.p75 + 1.5 * iqr
            extreme_threshold = val_stats.p75 + 3 * iqr
            
            print(f"  • IQR: {iqr:.4f}")
            print(f"  • Limite outliers (Q3 + 1.5×IQR): {outlier_threshold:.4f}")
            print(f"  • Limite extremos (Q3 + 3×IQR): {extreme_threshold:.4f}")
            
            # Contar outliers
            n_outliers = np.sum(concatenated_val > outlier_threshold)
            n_extremes = np.sum(concatenated_val > extreme_threshold)
            pct_outliers = (n_outliers / len(concatenated_val)) * 100
            pct_extremes = (n_extremes / len(concatenated_val)) * 100
            
            print(f"  • Outliers (>{outlier_threshold:.2f}): {n_outliers} ({pct_outliers:.2f}%)")
            print(f"  • Extremos (>{extreme_threshold:.2f}): {n_extremes} ({pct_extremes:.2f}%)")
            print()
            
            # Distribuição
            print("📈 DISTRIBUIÇÃO:")
            print(f"  • 50% dos valores: [{val_stats.p25:.2f}, {val_stats.p75:.2f}]")
            print(f"  • 80% dos valores: [{val_stats.p10:.2f}, {val_stats.p90:.2f}]")
            print(f"  • 90% abaixo de: {val_stats.p90:.2f}")
            print(f"  • 95% abaixo de: {val_stats.p95:.2f}")
            print(f"  • 99% abaixo de: {val_stats.p99:.2f}")
            print()
            
            # Diagnóstico
            ratio = val_stats.mean / val_stats.median if val_stats.median > 0 else 0
            
            print("⚠️  DIAGNÓSTICO:")
            if ratio > 3:
                print(f"  • Razão Média/Mediana: {ratio:.2f}x - MUITOS OUTLIERS!")
                print(f"  • A mediana ({val_stats.median:.2f}) é mais representativa que a média ({val_stats.mean:.2f})")
                print(f"  • Recomendação: Investigar {n_extremes} casos extremos (>{extreme_threshold:.2f})")
            elif ratio > 1.5:
                print(f"  • Razão Média/Mediana: {ratio:.2f}x - Distribuição assimétrica")
                print(f"  • Considere usar mediana como métrica principal")
            else:
                print(f"  • Razão Média/Mediana: {ratio:.2f}x - Distribuição razoável")
            
            print()
        
        print("=" * 80)
        print()
    
    def _train_single_window(
        self,
        model_builder: ModelBuilder,
        work_dir: Path,
        idx_train: int,
        train_data: Union[Window, SlicedDataset],
        incremental: bool,
        force_reset: bool,
        verbose: bool,
        save_model: bool,
        fit_kwargs: Dict[str, Any]
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Treina modelo em uma única janela"""
        self._print_window_header(idx_train, train_data)
        
        # Tentar carregar checkpoint
        model = None
        if not force_reset:
            model = model_builder.load_from_checkpoint(work_dir, idx_train)
        
        # Criar e treinar
        if model is None:
            model = model_builder.build(
                work_dir,
                idx_train,
                incremental,
                force_reset
            )
            
            print(f"🔧 Treinando modelo...")
            
            try:
                # Preparar argumentos
                fit_args = {
                    'series': train_data.series,
                    'past_covariates': train_data.past_covariates if model.supports_past_covariates else None,
                    'future_covariates': train_data.future_covariates if model.supports_future_covariates else None,
                    **fit_kwargs
                }
                
                # Validação para PyTorch
                if model_builder.is_torch_model:
                    fit_args.update({
                        'val_series': self._val_data['series'],
                        'val_past_covariates': self._val_data['past_covariates'] if model.supports_past_covariates else None,
                        'val_future_covariates': self._val_data['future_covariates'] if model.supports_future_covariates else None,
                    })
                
                # Verbose
                if 'verbose' in model.fit.__code__.co_varnames:
                    fit_args['verbose'] = verbose
                
                model.fit(**fit_args)
                print(f"✓ Treinamento concluído")
            
            except Exception as e:
                print(f"❌ Erro no treinamento da janela {idx_train}: {e}")
                raise
        
        # Avaliar métricas
        print(f"📈 Avaliando métricas...")
        train_metrics, val_window_metrics = self._eval_metrics(model, train_data)
        
        # Estatísticas
        train_stats = MetricStats.from_array(train_metrics)
        val_stats_list = [MetricStats.from_array(vm) for vm in val_window_metrics]
        
        # Imprimir
        self._print_metrics(train_stats, val_stats_list)
        
        # TensorBoard
        self._log_metrics_to_tensorboard(
            model, idx_train, train_stats, val_stats_list, model_builder.is_torch_model
        )
        
        # Salvar
        if save_model:
            model_builder.save_model(model, work_dir, idx_train)
            self._save_model_onnx(model, work_dir, idx_train, model_builder.is_torch_model)
            self._save_model_standard_format(
                model, 
                work_dir, 
                idx_train,
                model_builder.name,
                model_builder.is_torch_model
            )
            print()
        
        return train_metrics, val_window_metrics
    
    def fit(
        self,
        model_builder: ModelBuilder,
        metric_debug_fn: Optional[Callable] = None,
        incremental: bool = False,
        verbose: bool = False,
        save_model: bool = True,
        force_reset: bool = False,
        **kwargs
    ) -> Tuple[List[np.ndarray], List[List[np.ndarray]]]:
        """
        Treina modelo em múltiplas janelas
        
        Returns:
        --------
        tuple
            (train_metrics_list, val_metrics_by_window)
            - train_metrics_list: lista de arrays (um por janela de treino)
            - val_metrics_by_window: lista de listas de arrays
        """
        # Avisos
        if not model_builder.is_torch_model:
            if incremental:
                warnings.warn("Treinamento incremental não suportado para modelos não-PyTorch")
            if force_reset:
                warnings.warn("Parâmetro 'force_reset' não aplicável para modelos não-PyTorch")
        
        # Inicializar
        train_metrics_list = []
        val_metrics_by_window = []
        
        work_dir = self.get_train_work_dir(model_builder.name)
        
        # Cabeçalho
        self._print_training_header(model_builder.name, work_dir, model_builder.is_torch_model)
        
        # Treinar cada janela
        for idx_train, train_data in enumerate(self.train_data):
            train_metrics, val_window_metrics = self._train_single_window(
                model_builder=model_builder,
                work_dir=work_dir,
                idx_train=idx_train,
                train_data=train_data,
                incremental=incremental if model_builder.is_torch_model else False,
                force_reset=force_reset if model_builder.is_torch_model else False,
                verbose=verbose,
                save_model=save_model,
                fit_kwargs=kwargs
            )
            
            train_metrics_list.append(train_metrics)
            val_metrics_by_window.append(val_window_metrics)
            
            # Callback de debug
            if metric_debug_fn is not None:
                metric_debug_fn(
                    idx_train,
                    train_metrics_list,
                    val_metrics_by_window,
                    metric_name=self.eval_metric_fn.__name__
                )
        
        # Agregar todas as métricas
        all_train_metrics = np.concatenate(train_metrics_list)
        
        # Resumo final
        self._print_summary(work_dir, all_train_metrics, val_metrics_by_window)
        
        return train_metrics_list, val_metrics_by_window
    
    def __repr__(self) -> str:
        return (
            f"TrainPipeline(\n"
            f"  train_data={len(self.train_data)},\n"
            f"  val_data={len(self.val_data)},\n"
            f"  history_size={self.history_size},\n"
            f"  forecast_horizon={self.forecast_horizon},\n"
            f"  experiments_dir='{self.experiments_dir}',\n"
            f"  experiment_label='{self.experiment_label}',\n"
            f"  eval_metric={self.eval_metric_fn.__name__}\n"
            f")"
        )
import numpy as np

import random
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional

from darts.models import (
    DLinearModel,
    NLinearModel,
    TSMixerModel,
    TCNModel,
    NBEATSModel,
    LinearRegressionModel,
    NHiTSModel,
    LightGBMModel,
    XGBModel
)

from moviasai.dataset_loader import DatasetLoader
from moviasai.vehicle_profile import VehicleProfile

from moviasai.covariates import default_encoders
from moviasai.train import TrainPipeline, ModelBuilder, TrainerOptionsBuilder

from darts.metrics import wmape
from darts.utils.likelihood_models import QuantileRegression


class Pipeline:
    """
    Pipeline completo de treinamento para múltiplos datasets e modelos
    
    Esta classe gerencia o treinamento de múltiplos modelos em múltiplos
    datasets, incluindo configuração de seeds, carregamento de dados,
    extração de covariáveis e treinamento com logging.
    
    Suporta modelos PyTorch e Globais do Darts.
    
    Attributes
    ----------
    data_dirs : List[Path]
        Lista de diretórios com datasets
    work_dir : Path
        Diretório para salvar modelos e experimentos
    seed : int
        Seed para reprodutibilidade
    history_size : int
        Tamanho do histórico (inferido dos datasets)
    forecast_horizon : int
        Horizonte de previsão (inferido dos datasets)
    
    Examples
    --------
    >>> pipeline = CompletePipeline(
    ...     data_dirs=['../dataset/windows/dataset1', '../dataset/windows/dataset2'],
    ...     work_dir='../models',
    ...     seed=13
    ... )
    >>> pipeline.run()
    """
    
    def __init__(
        self,
        data_dirs: List[str],
        profile_dir: str = '../profile',
        work_dir: str = '../models',
        seed: int = 13,
        cache_dir: str = '../dataset/.cache',
        eval_metric_fn: Any = wmape,
        verbose: bool = True
    ):
        """
        Inicializa o pipeline completo
        
        Parameters
        ----------
        data_dirs : List[str]
            Lista de caminhos para diretórios de datasets
        work_dir : str, default='../models'
            Diretório para salvar modelos e experimentos
        seed : int, default=13
            Seed para reprodutibilidade
        cache_dir : str, default='../dataset/.cache'
            Diretório para cache de janelas
        eval_metric_fn : callable, default=wmape
            Função de métrica de avaliação
        verbose : bool, default=True
            Se True, imprime informações detalhadas
        """
        self.data_dirs = [Path(d) for d in data_dirs]
        self.profile_dir = Path(profile_dir)
        self.work_dir = Path(work_dir)
        self.seed = seed
        self.cache_dir = cache_dir
        self.eval_metric_fn = eval_metric_fn
        self.verbose = verbose
        
        # Configurar seeds
        self._set_seeds()
        
        # Criar diretório de trabalho
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Armazenar datasets carregados
        self.datasets = []
        self.dataset_names = []
        self.train_datasets = []
        self.val_datasets = []

    def _set_seeds(self):
        """Configura seeds para reprodutibilidade"""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        
        if self.verbose:
            print(f"✓ Seeds configuradas: {self.seed}\n")

    def load_datasets(self):
        """
        Carrega todos os datasets e prepara janelas de treino e validação
        
        Returns
        -------
        self
            Retorna self para permitir method chaining
            
        Examples
        --------
        >>> pipeline.load_datasets()
        ================================================================================
        CARREGANDO DATASETS
        ================================================================================
        Dataset 1/3: dataset_km_dia_clean_class0_hist28_fh7
          ✓ Carregado: 480,501 registros (treino), 177,415 (validação)
        ...
        """
        if self.verbose:
            print("="*80)
            print("CARREGANDO DATASETS")
            print("="*80)
            print()


        profiles = {
            'km': VehicleProfile.load(self.profile_dir / 'km'),
            'h':  VehicleProfile.load(self.profile_dir / 'h')
        }
        
        for idx, data_dir in enumerate(self.data_dirs, 1):
            if self.verbose:
                print(f"Dataset {idx}/{len(self.data_dirs)}: {data_dir.name}")
            
            # Nome do dataset
            dataset_name = data_dir.name
            self.dataset_names.append(dataset_name)
            
            # Criar extratores de covariáveis
            # ma = MovingAvg('ma', history_size=dataset.history_size, freq=7)

            metric = data_dir.name.split('_')[1]
            # Criar window loader
            dataset_loader = DatasetLoader(
                dataset_dir=data_dir, 
                vp=profiles[metric],
                # past_covariates_extractor=[ma],
            )

            self.datasets.append(dataset_loader)

            if self.verbose:
                print(f"  Target: {dataset_loader.target}")
                print(f"  History: {dataset_loader.history_size}d, Forecast: {dataset_loader.forecast_horizon}d")
            
            # Carregar janelas
            train_dataset = dataset_loader.load(
                'train',
                return_windows=False, 
                use_cache=True, 
                cache_dir=self.cache_dir
            )
            val_windows = dataset_loader.load(
                'val',
                return_windows=True,
                use_cache=True, 
                cache_dir=self.cache_dir
            )
            
            if self.verbose:
                print(f"  Janelas treino: {len(train_dataset.windows)}, validação: {len(val_windows)}")
            
            # Criar SlicedDataset para treino
            self.train_datasets.append(train_dataset)
            
            self.val_datasets.append(val_windows)
            
            if self.verbose:
                print(f"  ✓ Séries treino: {len(train_dataset)}, validação: {sum(len(w) for w in val_windows)}")
                print()
        
        if self.verbose:
            print("="*80)
            print(f"✓ {len(self.datasets)} datasets carregados")
            print("="*80)
            print()
        
        return self
    
    def _create_model_builders(self, likelihood: bool=False) -> Dict[str, ModelBuilder]:
        """
        Cria builders para todos os modelos
        
        Returns
        -------
        Dict[str, ModelBuilder]
            Dicionário {nome_modelo: builder}
        """
        # Inferir parâmetros do primeiro dataset
        dataset = self.datasets[0]
        history_size = dataset.history_size
        forecast_horizon = dataset.forecast_horizon
        
        # Opções do trainer para modelos PyTorch
        options_builder = TrainerOptionsBuilder(
            early_stopper_args={
                "monitor": "val_loss",
                "patience": 5,
                "min_delta": 0.0,
                "mode": "min"
            },
            add_progress_bar=True,
            val_check_interval=None,
            # limit_train_batches=0.1,
        )
        
        # Configurar modelos
        models = {}
        
        # ====================================================================
        # MODELOS PYTORCH
        # ====================================================================

        quantiles = MAINTENANCE_QUANTILES_STANDARD = [
            0.05,  # ↔ 0.95  (IC 90%)
            0.10,  # ↔ 0.90  (IC 80%)
            0.25,  # ↔ 0.75  (IC 50%)
            0.50,  # Mediana
            0.75,  # Par de 0.25
            0.90,  # Par de 0.10
            0.95,  # Par de 0.05
        ]

        # Configuração IDEAL para treino por grupos homogêneos:

        models['DLinear'] = ModelBuilder(
            model_cls=DLinearModel,
            options_builder=options_builder,
            input_chunk_length=history_size,
            output_chunk_length=forecast_horizon,
            kernel_size=25,
            shared_weights=True,      
            const_init=True,
            n_epochs=100,
            batch_size=64,
            likelihood=QuantileRegression(quantiles=quantiles) if likelihood else None,
            save_checkpoints=True,
            random_state=self.seed
        )

        models['TSMixer'] = ModelBuilder(
            model_cls=TSMixerModel,
            options_builder=options_builder,
            input_chunk_length=history_size,
            output_chunk_length=forecast_horizon,
            add_encoders=default_encoders(False),
            hidden_size=64,
            ff_size=64,
            num_blocks=2,
            dropout=0.1,
            activation='ReLU',
            norm_type='LayerNorm',
            n_epochs=100,
            batch_size=64,
            likelihood=QuantileRegression(quantiles=quantiles) if likelihood else None,
            save_checkpoints=True,
            random_state=self.seed
        )

        models['NLinear'] = ModelBuilder(
            model_cls=NLinearModel,
            options_builder=options_builder,
            input_chunk_length=history_size,
            output_chunk_length=forecast_horizon,
            shared_weights=True,     
            const_init=True,
            normalize=False,           
            n_epochs=100,
            batch_size=64,
            likelihood=QuantileRegression(quantiles=quantiles) if likelihood else None,
            save_checkpoints=True,
            random_state=self.seed
        )
        
        # TCN
        models['TCN'] = ModelBuilder(
            model_cls=TCNModel,
            options_builder=options_builder,
            input_chunk_length=history_size,
            output_chunk_length=forecast_horizon,
            add_encoders=default_encoders(False),
            n_epochs=100,
            batch_size=64,
            kernel_size=7,
            num_filters=128,
            dilation_base=3,
            dropout=0.2,
            likelihood=QuantileRegression(quantiles=quantiles) if likelihood else None,
            save_checkpoints=True,
            random_state=self.seed
        )
        
        # NBEATS
        models['NBEATS'] = ModelBuilder(
            model_cls=NBEATSModel,
            options_builder=options_builder,
            input_chunk_length=history_size,
            output_chunk_length=forecast_horizon,
            add_encoders=default_encoders(False),
            n_epochs=100,
            batch_size=64,
            num_stacks=30,
            num_blocks=1,
            num_layers=4,
            layer_widths=64,
            dropout=0.15,
            expansion_coefficient_dim=5,
            likelihood=QuantileRegression(quantiles=quantiles) if likelihood else None,
            save_checkpoints=True,
            random_state=self.seed
        )
        
        # NHiTS
        models['NHiTS'] = ModelBuilder(
            model_cls=NHiTSModel, 
            options_builder=options_builder,
            input_chunk_length=history_size,
            output_chunk_length=forecast_horizon,
            add_encoders=default_encoders(False),
            num_stacks=3,
            num_blocks=1,
            num_layers=4,
            layer_widths=512,
            n_epochs=100,
            batch_size=64,
            dropout=0.15,
            likelihood=QuantileRegression(quantiles=quantiles) if likelihood else None,
            save_checkpoints=True,
            random_state=self.seed
        )
        
        # ====================================================================
        # MODELOS GLOBAIS
        # ====================================================================
        
        # LightGBM
        models['LightGBM'] = ModelBuilder(
            model_cls=LightGBMModel,
            lags=history_size,
            output_chunk_length=forecast_horizon,
            lags_past_covariates=history_size,
            add_encoders=default_encoders(False), 
            # lags_future_covariates=list(range(-history_size, 0, 1)) + list(range(forecast_horizon)),
            num_leaves=31,
            learning_rate=0.1,
            n_estimators=100,
            random_state=self.seed
        )
        
        # XGBoost
        models['XGBoost'] = ModelBuilder(
            model_cls=XGBModel,
            lags=history_size,
            output_chunk_length=forecast_horizon,
            lags_past_covariates=history_size,
            add_encoders=default_encoders(False), 
            # lags_future_covariates=list(range(-history_size, 0, 1)) + list(range(forecast_horizon)),
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            random_state=self.seed
        )

        models['LinearRegression'] = ModelBuilder(
            model_cls=LinearRegressionModel,
            lags=history_size, 
            lags_past_covariates=history_size, 
            lags_future_covariates=None, 
            output_chunk_length=forecast_horizon, 
            add_encoders=default_encoders(False), 
            random_state=self.seed, 
            multi_models=True
        )
        
        return models
    
    def run(
        self,
        models_to_train: Optional[List[str]] = None,
        likelihood: Optional[bool] = True,
        incremental: bool = False,
        verbose_training: bool = True,
        save_models: bool = True,
        force_reset: bool = True,
        **fit_kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        Executa pipeline completo de treinamento
        
        Parameters
        ----------
        models_to_train : List[str], optional
            Lista de nomes de modelos a treinar.
            Se None, treina todos. Opções: 'TCN', 'NBEATS', 'RNN', 'NHiTS', 
            'LightGBM', 'XGBoost'
        likelihood : bool, default=True
            Se True, usa treinamento com quantiles (apenas PyTorch)
        incremental : bool, default=False
            Se True, usa treinamento incremental (apenas PyTorch)
        verbose_training : bool, default=True
            Se True, imprime logs detalhados durante treinamento
        save_models : bool, default=True
            Se True, salva modelos em disco
        force_reset : bool, default=True
            Se True, força reset dos modelos (apenas PyTorch)
        **fit_kwargs : dict
            Argumentos adicionais para o método fit
        
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Resultados estruturados:
            {
                'dataset_name': {
                    'model_name': {
                        'train_metrics': [...],
                        'val_metrics': [...]
                    }
                }
            }
        
        Examples
        --------
        Treinar todos os modelos em todos os datasets:
        
        >>> pipeline = CompletePipeline(data_dirs, work_dir='../models')
        >>> pipeline.load_datasets()
        >>> results = pipeline.run()
        
        Treinar apenas modelos específicos:
        
        >>> results = pipeline.run(models_to_train=['TCN', 'LightGBM'])
        
        Treinar com configurações customizadas:
        
        >>> results = pipeline.run(
        ...     incremental=True,
        ...     verbose_training=False,
        ...     stride=7,
        ...     load_best=True
        ... )
        """
        # Carregar datasets se ainda não carregados
        if not self.datasets:
            self.load_datasets()
        
        # Criar builders de modelos
        all_model_builders = self._create_model_builders(likelihood=likelihood)
        
        # Filtrar modelos a treinar
        if models_to_train is not None:
            model_builders = {
                name: builder 
                for name, builder in all_model_builders.items() 
                if name in models_to_train
            }
        else:
            model_builders = all_model_builders
        
        if self.verbose:
            print("="*80)
            print("CONFIGURAÇÃO DO TREINAMENTO")
            print("="*80)
            print(f"Datasets: {len(self.datasets)}")
            print(f"Modelos: {len(model_builders)} ({', '.join(model_builders.keys())})")
            print(f"Seed: {self.seed}")
            print(f"Work dir: {self.work_dir}")
            print("="*80)
            print()
        
        # Estrutura para armazenar resultados
        results = {}
        
        # Iterar sobre datasets
        for dataset_name, train_data, val_data, dataset in zip(
            self.dataset_names, 
            self.train_datasets, 
            self.val_datasets,
            self.datasets
        ):
            if self.verbose:
                print("\n" + "="*80)
                print(f"DATASET: {dataset_name}")
                print("="*80)
                print()
            
            results[dataset_name] = {}
            
            # Criar pipeline para este dataset
            pipeline = TrainPipeline(
                train_data=train_data,
                val_data=val_data,
                history_size=dataset.history_size,
                forecast_horizon=dataset.forecast_horizon,
                experiment_label=dataset_name,
                experiments_dir=str(self.work_dir),
                eval_metric_fn=self.eval_metric_fn
            )
            
            # Treinar cada modelo
            for model_name, model_builder in model_builders.items():
                if self.verbose:
                    print("\n" + "-"*80)
                    print(f"MODELO: {model_name}")
                    print("-"*80)
                    print()
                
                try:
                    # Treinar
                    train_metrics, val_metrics = pipeline.fit(
                        model_builder=model_builder,
                        metric_debug_fn=None,
                        incremental=incremental,
                        verbose=verbose_training,
                        save_model=save_models,
                        force_reset=force_reset,
                        **fit_kwargs
                    )
                    
                    # Armazenar resultados
                    results[dataset_name][model_name] = {
                        'train_metrics': train_metrics,
                        'val_metrics': val_metrics,
                        'status': 'success'
                    }
                    
                except Exception as e:
                    print(f"❌ Erro ao treinar {model_name}: {e}")
                    results[dataset_name][model_name] = {
                        'status': 'failed',
                        'error': str(e)
                    }
        
        if self.verbose:
            self._print_final_summary(results)
        
        return results
    
    def _print_final_summary(self, results: Dict[str, Dict[str, Any]]):
        """
        Imprime resumo final de todos os treinamentos
        
        Parameters
        ----------
        results : dict
            Resultados estruturados do treinamento
        """
        print("\n" + "="*80)
        print("RESUMO FINAL")
        print("="*80)
        print()
        
        for dataset_name, dataset_results in results.items():
            print(f"Dataset: {dataset_name}")
            
            for model_name, model_results in dataset_results.items():
                status = model_results['status']
                
                if status == 'success':
                    # Calcular métricas agregadas
                    all_train = np.concatenate(model_results['train_metrics'])
                    all_val = []
                    for val_window_metrics in model_results['val_metrics']:
                        for val_metrics in val_window_metrics:
                            all_val.append(val_metrics)
                    all_val = np.concatenate(all_val)
                    
                    train_mean = np.mean(all_train[np.isfinite(all_train)])
                    val_mean = np.mean(all_val[np.isfinite(all_val)])
                    
                    print(f"  ✓ {model_name:12s}: Train={train_mean:.4f}, Val={val_mean:.4f}")
                else:
                    print(f"  ✗ {model_name:12s}: {model_results.get('error', 'Erro desconhecido')}")
            
            print()
        
        print("="*80)
        print()
    def __repr__(self) -> str:
        """Representação em string do objeto"""
        return (
            f"CompletePipeline(\n"
            f"  n_datasets={len(self.data_dirs)},\n"
            f"  work_dir='{self.work_dir}',\n"
            f"  seed={self.seed},\n"
            f"  datasets_loaded={len(self.datasets)}\n"
            f")"
        )

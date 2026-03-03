# api/services/training_service.py

from typing import Dict
from pathlib import Path
import logging
import pickle
from sqlalchemy.ext.asyncio import AsyncSession

from api.config.training_config import TrainingConfig
from moviasai.train_utils import Pipeline

logger = logging.getLogger(__name__)


class TrainingService:
    """
    Service para treinamento de modelos
    
    Gerencia o treinamento de modelos PyTorch e Globais usando
    a classe Pipeline do train_utils.
    """
    
    def __init__(
        self,
        session: AsyncSession,
        config: TrainingConfig
    ):
        self.session = session
        self.config = config
    
    async def run_training(
        self,
        train_pytorch: bool = True,
        train_global: bool = True,
        verbose: bool = True
    ) -> Dict[str, any]:
        """
        Executa treinamento de modelos
        
        Parameters
        ----------
        train_pytorch : bool, default=True
            Se True, treina modelos PyTorch
        train_global : bool, default=True
            Se True, treina modelos Globais
        verbose : bool, default=True
            Se True, exibe logs detalhados
        
        Returns
        -------
        dict
            Resultados do treinamento:
            - pytorch_results: resultados dos modelos PyTorch
            - global_results: resultados dos modelos Globais
            - models_trained: lista de modelos treinados
            - work_dir: diretório de trabalho
        """
        logger.info("="*60)
        logger.info("INICIANDO TREINAMENTO DE MODELOS")
        logger.info("="*60)
        
        # Preparar diretórios de dados
        windows_path = Path(self.config.windows_dir)
        data_dirs = [
            p for p in windows_path.iterdir() 
            if p.is_dir() and p.stem.startswith('dataset')
        ]
        
        if not data_dirs:
            raise FileNotFoundError(
                f"Nenhum dataset encontrado em {windows_path}"
            )
        
        logger.info(f"Datasets encontrados: {len(data_dirs)}")
        for d in data_dirs:
            logger.info(f"  - {d.name}")
        
        # Criar pipeline
        pipeline = Pipeline(
            data_dirs=[str(d) for d in data_dirs],
            profile_dir=self.config.profile_dir,
            work_dir=self.config.work_dir,
            seed=self.config.seed,
            cache_dir=self.config.cache_dir,
            verbose=verbose
        )
        
        # Carregar datasets
        logger.info("\nCarregando datasets...")
        pipeline.load_datasets()
        
        results = {
            'pytorch_results': None,
            'global_results': None,
            'models_trained': [],
            'work_dir': str(self.config.work_dir)
        }
        
        # Treinar modelos PyTorch
        if train_pytorch:
            pytorch_models = self.config.get_pytorch_models_to_train()
            
            if pytorch_models:
                logger.info(f"\n{'='*60}")
                logger.info("TREINANDO MODELOS PYTORCH")
                logger.info(f"{'='*60}")
                logger.info(f"Modelos: {pytorch_models}")
                logger.info(f"Likelihood: {self.config.pytorch.likelihood}")
                logger.info(f"Incremental: {self.config.pytorch.incremental}")
                logger.info(f"Epochs: {self.config.pytorch.n_epochs}")
                logger.info(f"Batch size: {self.config.pytorch.batch_size}")
                
                try:
                    results['pytorch_results'] = pipeline.run(
                        models_to_train=pytorch_models,
                        likelihood=self.config.pytorch.likelihood,
                        incremental=self.config.pytorch.incremental,
                        verbose_training=self.config.options.verbose_training,
                        save_models=self.config.options.save_models,
                        force_reset=self.config.pytorch.force_reset,
                        load_best=self.config.pytorch.load_best
                    )
                    
                    results['models_trained'].extend(pytorch_models)
                    
                    # Salvar resultados
                    if self.config.options.save_results:
                        self._save_results(
                            results['pytorch_results'],
                            'training_results_pytorch.pkl'
                        )
                    
                    logger.info("✅ Modelos PyTorch treinados com sucesso")
                    
                except Exception as e:
                    logger.error(f"❌ Erro ao treinar modelos PyTorch: {e}")
                    logger.exception(e)
                    results['pytorch_error'] = str(e)
            else:
                logger.info("⏭️  Modelos PyTorch desabilitados")
        
        # Treinar modelos Globais
        if train_global:
            global_models = self.config.get_global_models_to_train()
            
            if global_models:
                logger.info(f"\n{'='*60}")
                logger.info("TREINANDO MODELOS GLOBAIS")
                logger.info(f"{'='*60}")
                logger.info(f"Modelos: {global_models}")
                logger.info(f"Stride: {self.config.global_.stride}")
                
                try:
                    results['global_results'] = pipeline.run(
                        models_to_train=global_models,
                        likelihood=False,  # Modelos globais não usam likelihood
                        incremental=False,
                        verbose_training=self.config.options.verbose_training,
                        save_models=self.config.options.save_models,
                        stride=self.config.global_.stride
                    )
                    
                    results['models_trained'].extend(global_models)
                    
                    # Salvar resultados
                    if self.config.options.save_results:
                        self._save_results(
                            results['global_results'],
                            'training_results_global.pkl'
                        )
                    
                    logger.info("✅ Modelos Globais treinados com sucesso")
                    
                except Exception as e:
                    logger.error(f"❌ Erro ao treinar modelos Globais: {e}")
                    logger.exception(e)
                    results['global_error'] = str(e)
            else:
                logger.info("⏭️  Modelos Globais desabilitados")
        
        logger.info(f"\n{'='*60}")
        logger.info("TREINAMENTO CONCLUÍDO")
        logger.info(f"{'='*60}")
        logger.info(f"Modelos treinados: {results['models_trained']}")
        logger.info(f"Diretório: {results['work_dir']}")
        
        return results
   
    def _save_results(self, results: dict, filename: str):
        """Salva resultados em arquivo pickle"""
        results_path = Path(self.config.work_dir) / filename
        
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"✅ Resultados salvos: {results_path}")
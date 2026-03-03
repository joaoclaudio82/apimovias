# api/services/data_ingestion_service.py

from typing import Dict
from pathlib import Path
import logging
import shutil
import pandas as pd
import polars as pl
from sqlalchemy.ext.asyncio import AsyncSession

from api.config.data_ingestion_config import DataIngestionConfig
from api.config.vehicle_profile_config import VehicleProfileConfig
from api.services.vehicle_profile_service import VehicleProfileService
from moviasai.dataset_utils import DatasetFormatter, DatasetGenerator, ClassSplitter
from moviasai.vehicle_profile import VehicleProfile

logger = logging.getLogger(__name__)


class DataIngestionService:
    """
    Service para ingestão e processamento de dados de telemetria
    
    Fluxo:
    1. Obter segmentação do banco
    2. Gerar perfis versionados (VehicleProfile)
    3. Formatar dataset (DatasetFormatter)
    4. Dividir por segmento (ClassSplitter)
    5. Gerar janelas (DatasetGenerator)
    6. Limpar arquivos intermediários (opcional)
    """
    
    def __init__(
        self,
        session: AsyncSession,
        config: DataIngestionConfig,
        profile_config: VehicleProfileConfig 
    ):
        self.session = session
        self.config = config
        self.profile_config = profile_config
        self.profile_service = VehicleProfileService(session, profile_config, config)
    
    async def run_ingestion(
        self,
        clean_intermediate: bool = False,
        verbose: bool = True
    ) -> Dict[str, any]:
        """
        Executa pipeline completo de ingestão
        
        Parameters
        ----------
        clean_intermediate : bool, default=False
            Se True, remove arquivos intermediários (mantém só windows e profiles)
        verbose : bool, default=True
            Se True, exibe logs detalhados
        
        Returns
        -------
        dict
            Estatísticas da ingestão
        """
        logger.info("="*60)
        logger.info("INICIANDO INGESTÃO DE DADOS")
        logger.info("="*60)
        
        stats = {
            'categories_processed': [],
            'total_windows_generated': 0,
            'output_dir': str(self.config.get_windows_dir()),
            'profile_dir': str(self.config.get_profile_dir()), 
            'intermediate_cleaned': clean_intermediate
        }
        
        # 1. Obter segmentação do banco
        logger.info(f"\n{'='*60}")
        logger.info("1️⃣ OBTENDO SEGMENTAÇÃO DO BANCO")
        logger.info(f"{'='*60}")
        
        df_km = await self.profile_service.get_vehicles_by_category('km')
        df_h = await self.profile_service.get_vehicles_by_category('h')
        
        logger.info(f"✅ Veículos KM: {len(df_km)}")
        logger.info(f"✅ Veículos H: {len(df_h)}")
        
        # 2. Gerar perfis versionados
        logger.info(f"\n{'='*60}")
        logger.info("2️⃣ GERANDO PERFIS VERSIONADOS")
        logger.info(f"{'='*60}")
        
        profiles = await self._generate_profiles(df_km, df_h, verbose)
        
        logger.info(f"✅ Perfis gerados e salvos em: {stats['profile_dir']}")
        
        # 3. Processar cada categoria
        for category in self.config.categories:
            logger.info(f"\n{'='*60}")
            logger.info(f"3️⃣ Processando categoria: {category}")
            logger.info(f"{'='*60}")
            
            try:
                metric = category.split('_')[0]
                df_segments = df_km if metric == 'km' else df_h
                
                result = await self._process_category(
                    category,
                    df_segments,
                    verbose
                )
                
                stats['categories_processed'].append(category)
                stats[f'{category}_windows'] = result['n_windows']
                stats['total_windows_generated'] += result['n_windows']
                
                logger.info(f"✅ Categoria {category} processada: {result['n_windows']} janelas")
                
            except Exception as e:
                logger.error(f"❌ Erro ao processar {category}: {e}")
                logger.exception(e)
                stats[f'{category}_error'] = str(e)
        
        # 4. Limpar intermediários
        if clean_intermediate:
            logger.info(f"\n{'='*60}")
            logger.info("4️⃣ Limpando arquivos intermediários...")
            logger.info(f"{'='*60}")
            
            self._clean_intermediate_files()
            logger.info("✅ Arquivos intermediários removidos")
        
        logger.info(f"\n{'='*60}")
        logger.info("INGESTÃO CONCLUÍDA")
        logger.info(f"{'='*60}")
        logger.info(f"Categorias processadas: {len(stats['categories_processed'])}")
        logger.info(f"Total de janelas: {stats['total_windows_generated']}")
        logger.info(f"Perfis salvos em: {stats['profile_dir']}")
        logger.info(f"Janelas salvas em: {stats['output_dir']}")
        
        return stats

    async def _generate_profiles(
        self,
        df_km: pd.DataFrame,
        df_h: pd.DataFrame,
        verbose: bool
    ) -> Dict[str, VehicleProfile]:
        """Gera perfis versionados para KM e H"""
        
        profile_dir = self.config.get_profile_dir()  
        profile_dir.mkdir(parents=True, exist_ok=True)
        
        # Carregar dataset completo
        logger.info(f"Carregando dataset completo: {self.config.dataset_path}")
        
        df_full = pl.read_csv(
            self.config.dataset_path,
            columns=['veiculo_id', 'data', 'km_dia_clean', 'h_dia_clean'],
            try_parse_dates=True
        )
        
        logger.info(f"✅ Dataset carregado: {len(df_full)} registros")
        
        profiles = {}
        
        # Gerar perfil KM
        logger.info(f"\n📊 Gerando perfil KM...")
        
        df_full_km = df_full.filter(
            pl.col("veiculo_id").is_in(df_km['veiculo_id'].tolist())
        )
        
        logger.info(f"   Veículos KM: {len(df_km)}")
        logger.info(f"   Registros KM: {len(df_full_km)}")
        
        vp_km = VehicleProfile(
            target='km_dia_clean',
            sample_size=self.profile_config.profile.sample_size,  
            p_upper=self.profile_config.profile.p_upper,          
            n_jobs=self.profile_config.profile.n_jobs             
        )
        
        vp_km.fit(df_full_km, verbose=verbose)
        
        profile_km_path = profile_dir / 'km'
        vp_km.save(profile_km_path)
        
        profiles['km'] = vp_km
        
        logger.info(f"✅ Perfil KM salvo: {profile_km_path}")
        
        # Gerar perfil H
        logger.info(f"\n📊 Gerando perfil H...")
        
        df_full_h = df_full.filter(
            pl.col("veiculo_id").is_in(df_h['veiculo_id'].tolist())
        )
        
        logger.info(f"   Veículos H: {len(df_h)}")
        logger.info(f"   Registros H: {len(df_full_h)}")
        
        vp_h = VehicleProfile(
            target='h_dia_clean',
            sample_size=self.profile_config.profile.sample_size,  
            p_upper=self.profile_config.profile.p_upper,          
            n_jobs=self.profile_config.profile.n_jobs             
        )
        
        vp_h.fit(df_full_h, verbose=verbose)
        
        profile_h_path = profile_dir / 'h'
        vp_h.save(profile_h_path)
        
        profiles['h'] = vp_h
        
        logger.info(f"✅ Perfil H salvo: {profile_h_path}")
        
        return profiles
    
    async def _process_category(
        self,
        category: str,
        df_segments: pd.DataFrame,
        verbose: bool
    ) -> Dict[str, any]:
        """Processa uma categoria (km ou h)"""
        
        metric = category.split('_')[0]
        
        # 1. Formatar dataset
        logger.info(f"1️⃣ Formatando dataset para {category}...")
        
        formatter = DatasetFormatter(
            dataset_path=self.config.dataset_path,
            target=category,
            min_days=self.config.formatter.min_days,
            min_weeks=self.config.formatter.min_weeks,
            max_gap=self.config.formatter.max_gap
        )
        
        output_dir_cat = self.config.get_output_dir_for_category(category)
        output_dir_cat.mkdir(parents=True, exist_ok=True)
        
        dataset_file = output_dir_cat / f"dataset_{category}_mindays{self.config.formatter.min_days}_minweeks{self.config.formatter.min_weeks}_maxgap{self.config.formatter.max_gap}.csv"
        
        # Formatar e salvar
        df_formatted = formatter.format()
        df_formatted.write_csv(dataset_file)
        
        logger.info(f"   ✅ Dataset formatado: {len(df_formatted)} registros")
        logger.info(f"   📁 Salvo em: {dataset_file}")
        
        # 2. Dividir por segmento
        logger.info(f"2️⃣ Dividindo por segmento...")
        
        segments_dir = self.config.get_segments_dir_for_category(category)
        segments_dir.mkdir(parents=True, exist_ok=True)
        
        splitter = ClassSplitter(df_segments, class_column='segment')
        
        splitter.split(
            data_path=dataset_file,
            output_dir=segments_dir,
            suffix='class'
        )
        
        # Verificar arquivos criados
        segment_files = list(segments_dir.glob("dataset_*.csv"))
        n_segments = len(segment_files)
        
        logger.info(f"   ✅ Arquivos de segmento criados: {n_segments}")
        for seg_file in segment_files:
            logger.info(f"      📄 {seg_file.name}")
        
        # 3. Gerar janelas
        logger.info(f"3️⃣ Gerando janelas...")
        
        windows_dir = self.config.get_windows_dir()
        windows_dir.mkdir(parents=True, exist_ok=True)
        
        n_windows = self._generate_windows(
            input_dir=segments_dir,
            output_dir=windows_dir,
            verbose=verbose
        )
        
        logger.info(f"   ✅ Janelas geradas: {n_windows}")
        
        return {
            'category': category,
            'n_records': len(df_formatted),
            'n_vehicles': len(df_segments),
            'n_segments': n_segments,
            'n_windows': n_windows
        }
    
    def _generate_windows(
        self,
        input_dir: Path,
        output_dir: Path,
        verbose: bool
    ) -> int:
        """Gera janelas para todos os arquivos de segmento"""
        
        files = list(input_dir.glob("dataset_*.csv"))
        total_windows = 0
        
        for file in files:
            logger.info(f"   📄 Processando: {file.name}")
            
            gen = DatasetGenerator(
                str(file),
                history_size=self.config.generator.history_size,
                forecast_horizon=self.config.generator.forecast_horizon
            )
            
            gen.generate(
                n_windows_test=self.config.generator.n_windows_test,
                n_windows_val=self.config.generator.n_windows_val,
                skip_n_days=self.config.generator.skip_n_days,
                use_effective_period=self.config.generator.use_effective_period,
                output_dir=str(output_dir),
                verbose=verbose
            )
            
            # Contar janelas geradas (assumindo 3 arquivos: train, val, test)
            total_windows += 3
        
        return total_windows
    
    def _clean_intermediate_files(self):
        """Remove arquivos intermediários (mantém só windows e profile)"""
        
        base_dir = Path(self.config.output_dir)
        
        # Remover diretórios km/ e h/ (exceto profile/)
        for metric in ['km', 'h']:
            metric_dir = base_dir / metric
            
            if metric_dir.exists():
                logger.info(f"   🗑️  Removendo: {metric_dir}")
                shutil.rmtree(metric_dir)
        
        logger.info(f"   ✅ Mantidos:")
        logger.info(f"      - {self.config.get_windows_dir()}")
        logger.info(f"      - {self.config.get_profile_dir()}")

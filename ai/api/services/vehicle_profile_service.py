# api/services/vehicle_profile_service.py

from typing import List, Dict, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from datetime import datetime
from pathlib import Path

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

import pandas as pd
import polars as pl
import numpy as np
import logging

from api.models import (
    Vehicle, 
    VehicleSummaryFeatures, 
    VehicleWeekdayFeatures,
    VehicleCategory
)
from moviasai.feature_extraction import (
    SegmentationFeatureExtractor,
    WeekdayFeatureExtractor
)
from moviasai.vehicle_profile import VehicleProfile
from moviasai.classification import VehiclePredictor
from moviasai.dataset_utils import DatasetFormatter, DatasetMerger

from api.config.vehicle_profile_config import VehicleProfileConfig
from api.config.data_ingestion_config import DataIngestionConfig


logger = logging.getLogger(__name__)


class VehicleProfileService:
    """Service para gerenciar perfis de veículos"""
    def __init__(
        self,
        session: AsyncSession,
        config: VehicleProfileConfig,
        ingestion_config: DataIngestionConfig
    ):
        self.session = session
        self.config = config
        self.ingestion_config = ingestion_config

    async def import_from_file(
        self,
        file_path: Optional[str] = None,
        vehicle_ids: Optional[List[int]] = None
    ) -> Dict[str, any]:
        """
        Importa e atualiza perfis de arquivo externo
        Fluxo completo:
        1. Carrega arquivo (usa config.input_data se não fornecido)
        2. Divide em veículos existentes (km/h) e novos (unk)
        3. Recupera samples do banco para veículos existentes
        4. Concatena dados novos com samples antigos
        5. Classifica veículos novos (category e segment)
        6. Reclassifica segments de todos (atualização)
        7. Extrai features completas
        8. Salva no banco
        Parameters
        ----------
        file_path : str, optional
            Path do arquivo. Se None, usa config.input_data
            vehicle_ids : List[int], optional
            Se fornecido, processa apenas esses veículos
        Returns
        -------
        dict
            Estatísticas da operação:
            - total_vehicles: total processado
            - existing_updated: veículos existentes atualizados
            - new_added: veículos novos adicionados
            - km_vehicles: total de veículos KM
            - h_vehicles: total de veículos H
        """

        logger.info("="*60)
        logger.info("IMPORTAÇÃO DE PERFIS")
        logger.info("="*60)
        
        # 1. Carregar arquivo
        if file_path is None:
            file_path = self.config.input_data
        
        logger.info(f"\n1️⃣ Carregando arquivo: {file_path}")
        df_raw = pl.read_csv(
            file_path,
            columns=['veiculo_id', 'data', 'h_dia_clean', 'km_dia_clean'],
            try_parse_dates=True
        )
        logger.info(f"✅ {len(df_raw):,} registros, {df_raw['veiculo_id'].n_unique()} veículos")
        
        # Filtrar vehicle_ids se fornecido
        if vehicle_ids is not None:
            df_raw = df_raw.filter(pl.col('veiculo_id').is_in(vehicle_ids))
            logger.info(f"  Filtrado para {len(vehicle_ids)} veículos específicos")
        
        # 2. Recuperar veículos existentes do banco
        logger.info(f"\n2️⃣ Recuperando veículos existentes do banco...")
        all_vehicle_ids = df_raw['veiculo_id'].unique().to_list()
        stmt = select(Vehicle).where(Vehicle.id.in_(all_vehicle_ids))
        result = await self.session.execute(stmt)
        
        existing_vehicles = {v.id: v for v in result.scalars().all()}
        logger.info(f"✅ {len(existing_vehicles)} veículos existentes")
        logger.info(f"  {len(all_vehicle_ids) - len(existing_vehicles)} veículos novos")
        
        # 3. Dividir em existentes (por category) e novos
        logger.info(f"\n3️⃣ Dividindo veículos por categoria...")
        
        existing_km_ids = [vid for vid, v in existing_vehicles.items() if v.category == VehicleCategory.KM]
        existing_h_ids = [vid for vid, v in existing_vehicles.items() if v.category == VehicleCategory.H]
        new_vehicle_ids = [vid for vid in all_vehicle_ids if vid not in existing_vehicles]
        
        df_km = df_raw.filter(pl.col('veiculo_id').is_in(existing_km_ids)).select(['veiculo_id', 'data', 'km_dia_clean'])
        df_h = df_raw.filter(pl.col('veiculo_id').is_in(existing_h_ids)).select(['veiculo_id', 'data', 'h_dia_clean'])
        df_unk = df_raw.filter(pl.col('veiculo_id').is_in(new_vehicle_ids))
        
        logger.info(f"✅ KM existentes: {len(existing_km_ids)}")
        logger.info(f"✅ H existentes: {len(existing_h_ids)}")
        logger.info(f"✅ Novos (a classificar): {len(new_vehicle_ids)}")
        
        # 4. Atualizar veículos existentes (concatenar com samples)
        logger.info(f"\n4️⃣ Atualizando veículos existentes...")
        df_km_updated, invalid_km_ids = await self._update_existing_vehicles(
            df_km, existing_km_ids, existing_vehicles, 'km_dia_clean', self.config.profile.sample_size
        )
        
        df_h_updated, invalid_h_ids = await self._update_existing_vehicles(
            df_h, existing_h_ids, existing_vehicles, 'h_dia_clean', self.config.profile.sample_size
        )
        
        all_invalid_ids = invalid_km_ids + invalid_h_ids
    
        if all_invalid_ids:
            logger.warning(f"\n🗑️  Excluindo {len(all_invalid_ids)} veículos que não satisfazem requisitos...")
            await self._delete_vehicles(all_invalid_ids)
            logger.info(f"  ✅ Veículos excluídos: {all_invalid_ids}")

        # ✅ VERIFICAR SE HÁ DADOS PARA PROCESSAR
        has_km_updates = len(df_km_updated) > 0
        has_h_updates = len(df_h_updated) > 0
        has_new_vehicles = len(new_vehicle_ids) > 0

        if not has_km_updates and not has_h_updates and not has_new_vehicles:
            logger.info("\n⏭️  NENHUM DADO NOVO ENCONTRADO")
            logger.info("  Todos os veículos já estão atualizados")
            logger.info("="*60)
            return {
                'total_vehicles': len(all_vehicle_ids),
                'existing_updated': 0,
                'new_added': 0,
                'km_vehicles': 0,
                'h_vehicles': 0,
                'message': 'Nenhum dado novo para processar'
            }
        
        # 5. Classificar veículos novos
        logger.info(f"\n5️⃣ Classificando veículos novos...")
        df_km_new, df_h_new = await self._classify_new_vehicles(df_unk, self.config.profile.sample_size)

        logger.info(f"✅ {len(df_km_new['veiculo_id'].unique()) if len(df_km_new) > 0 else 0} veículos classificados como KM")
        logger.info(f"✅ {len(df_h_new['veiculo_id'].unique()) if len(df_h_new) > 0 else 0} veículos classificados como H")
        
        # 6. Concatenar e limitar a sample_size
        logger.info(f"\n6️⃣ Consolidando dados finais...")
        df_km_final = self._concatenate(df_km_updated, df_km_new)
        df_h_final = self._concatenate(df_h_updated, df_h_new)
        
        # ✅ VERIFICAR NOVAMENTE SE HÁ DADOS FINAIS
        if len(df_km_final) == 0 and len(df_h_final) == 0:
            logger.info("\n⏭️  NENHUM DADO PARA PROCESSAR APÓS CONSOLIDAÇÃO")
            logger.info("="*60)
            return {
                'total_vehicles': len(all_vehicle_ids),
                'existing_updated': 0,
                'new_added': 0,
                'km_vehicles': 0,
                'h_vehicles': 0,
                'deleted_vehicles': len(all_invalid_ids),
                'message': 'Nenhum dado novo para processar'
            }
        
        logger.info(f"✅ KM final: {df_km_final['veiculo_id'].n_unique() if len(df_km_final) > 0 else 0} veículos")
        logger.info(f"✅ H final: {df_h_final['veiculo_id'].n_unique() if len(df_h_final) > 0 else 0} veículos")
        
        # 7. Reclassificar segments de TODOS os veículos (apenas os que têm dados)
        logger.info(f"\n7️⃣ Classificando segments...")
        all_segments = await self._classify_all_segments(df_km_final, df_h_final)
        
        # 8. Extrair features e salvar
        logger.info(f"\n8️⃣ Extraindo features e salvando...")
        saved_km = await self._extract_and_save(df_km_final, 'km', all_segments)
        saved_h = await self._extract_and_save(df_h_final, 'h', all_segments)

        n_updated = 0
        if len(df_km_updated) > 0:
            n_updated += df_km_updated['veiculo_id'].n_unique() 
        if len(df_h_updated) > 0:
            n_updated += df_h_updated['veiculo_id'].n_unique()

        n_new = 0
        if len(df_km_new) > 0:
            n_new += df_km_new['veiculo_id'].n_unique() 
        
        if len(df_h_new) > 0:
            n_new += df_h_new['veiculo_id'].n_unique()
        
        logger.info(f"\n✅ CONCLUÍDO")
        logger.info(f"  Total processado: {len(all_vehicle_ids)}")
        logger.info(f"  Existentes atualizados: {n_updated}")
        logger.info(f"  Novos adicionados: {n_new}")
        logger.info(f"  Veículos excluídos: {len(all_invalid_ids)}")
        logger.info(f"  KM: {saved_km}")
        logger.info(f"  H: {saved_h}")
        logger.info("="*60)

        return {
            'total_vehicles': len(all_vehicle_ids),
            'existing_updated': n_updated,
            'new_added': n_new,
            'deleted_vehicles': len(all_invalid_ids),
            'km_vehicles': saved_km,
            'h_vehicles': saved_h
        }

    async def _update_existing_vehicles(
        self,
        df_new: pl.DataFrame,
        vehicle_ids: List[int],
        existing_vehicles: Dict[int, Vehicle],
        target: str,
        sample_size: int
    ) -> Tuple[pl.DataFrame, List[int]]:
        """
        Atualiza dados de veículos existentes concatenando com samples
        
        Fluxo:
        - Recupera samples do banco
        - Filtra df_new para datas posteriores a samples_end_date
        - Descarta veículos sem dados novos
        - Concatena samples + novos dados
        - Preenche gaps com zero
        - Limita as séries ao máximo de sample_size registros
        - Valida requisitos mínimos (min_days, min_weeks, max_gap)
        - Retorna veículos válidos e lista de IDs para exclusão
        
        Returns
        -------
        Tuple[pl.DataFrame, List[int]]
            - DataFrame com veículos válidos
            - Lista de vehicle_ids que não satisfazem requisitos (para exclusão)
        """
        if not vehicle_ids:
            return pl.DataFrame(), []
        
        logger.info(f"  Atualizando {len(vehicle_ids)} veículos ({target})...")
        
        # 1. Criar DataFrame com samples do banco
        samples_data = []
        for vid in vehicle_ids:
            vehicle = existing_vehicles[vid]
            
            # Criar range de datas dos samples
            date_range = pl.date_range(
                vehicle.samples_start_date,
                vehicle.samples_end_date,
                interval='1d',
                eager=True
            )
            
            samples_data.append(
                pl.DataFrame({
                    'veiculo_id': [vid] * len(date_range),
                    'data': date_range,
                    target: vehicle.samples,
                    'samples_end_date': [vehicle.samples_end_date] * len(date_range)
                })
            )
        
        if not samples_data:
            return pl.DataFrame(), []
        
        df_samples = pl.concat(samples_data)
        
        # 2. Filtrar dados novos (posteriores a samples_end_date)
        df_new_with_end = df_new.join(
            df_samples.select(['veiculo_id', 'samples_end_date']).unique(),
            on='veiculo_id',
            how='left'
        )
        
        # Filtrar apenas datas posteriores
        df_new_filtered = df_new_with_end.filter(
            pl.col('data') > pl.col('samples_end_date')
        ).select(['veiculo_id', 'data', target])
        
        # 3. Descartar veículos sem dados novos
        vehicles_with_new_data = df_new_filtered['veiculo_id'].unique().to_list()
        
        if not vehicles_with_new_data:
            logger.info(f"  ⚠️  Nenhum veículo com dados novos, todos descartados")
            return pl.DataFrame(), []
        
        logger.info(f"  ✅ {len(vehicles_with_new_data)} veículos com dados novos")
        logger.info(f"  ⏭️  {len(vehicle_ids) - len(vehicles_with_new_data)} veículos sem dados novos (descartados)")
        
        # Filtrar samples apenas dos veículos com dados novos
        df_samples_filtered = df_samples.filter(
            pl.col('veiculo_id').is_in(vehicles_with_new_data)
        ).select(['veiculo_id', 'data', target])
        
        # 4. Concatenar samples + novos dados
        df_combined = pl.concat([df_samples_filtered, df_new_filtered])
        
        # 5. Preencher gaps com zero (vetorizado)
        date_ranges = (
            df_combined
            .group_by('veiculo_id')
            .agg([
                pl.col('data').min().alias('min_date'),
                pl.col('data').max().alias('max_date')
            ])
        )
        
        # Criar ranges completos para cada veículo
        full_ranges = []
        for row in date_ranges.iter_rows(named=True):
            vid = row['veiculo_id']
            min_date = row['min_date']
            max_date = row['max_date']
            
            date_range = pl.date_range(min_date, max_date, interval='1d', eager=True)
            
            full_ranges.append(
                pl.DataFrame({
                    'veiculo_id': [vid] * len(date_range),
                    'data': date_range
                })
            )
        
        df_full_ranges = pl.concat(full_ranges)
        
        # 6. Left join para preencher gaps
        df_filled = df_full_ranges.join(
            df_combined,
            on=['veiculo_id', 'data'],
            how='left'
        ).with_columns(
            pl.col(target).fill_null(0.0)
        ).sort(['veiculo_id', 'data'])
        
        logger.info(f"  ✅ {len(df_filled):,} registros após preenchimento de gaps")
        
        # 7. Limitar a sample_size
        df_limited = (
            df_filled
            .sort(['veiculo_id', 'data'])
            .group_by('veiculo_id')
            .tail(sample_size)
        )
        
        # 8. VALIDAR REQUISITOS MÍNIMOS
        logger.info(f"  🔍 Validando requisitos mínimos...")
        
        formatter = DatasetFormatter(
            df=df_limited.select(['veiculo_id', 'data', target]),
            target=target,
            min_days=self.ingestion_config.formatter.min_days,
            min_weeks=self.ingestion_config.formatter.min_weeks,
            max_gap=self.ingestion_config.formatter.max_gap
        )
        
        # Formatar (aplica filtros de qualidade)
        df_formatted = formatter.format(output_dir=None, verbose=False)
        
        # Veículos que passaram na validação
        valid_vehicle_ids = df_formatted['veiculo_id'].unique().to_list()
        
        # Veículos que falharam (devem ser excluídos)
        invalid_vehicle_ids = [vid for vid in vehicles_with_new_data if vid not in valid_vehicle_ids]
        
        logger.info(f"  ✅ {len(valid_vehicle_ids)} veículos válidos (satisfazem requisitos)")
        
        if invalid_vehicle_ids:
            logger.warning(f"  ❌ {len(invalid_vehicle_ids)} veículos NÃO satisfazem requisitos mínimos")
            logger.warning(f"     Esses veículos serão marcados para EXCLUSÃO: {invalid_vehicle_ids[:10]}...")
        
        return df_formatted, invalid_vehicle_ids

    async def _classify_new_vehicles(
        self,
        df_unk: pl.DataFrame,
        sample_size: int
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Classifica veículos novos em KM ou H
        
        Returns
        -------
        tuple
            (df_km_new, df_h_new)
        """
        if len(df_unk) == 0:
            return pl.DataFrame(), pl.DataFrame()
        
        df_limited = (
            df_unk
            .sort(['veiculo_id', 'data'])
            .group_by('veiculo_id')
            .tail(sample_size)
        )
        
        # Formatar dados para classificação
        formatter_km = DatasetFormatter(
            df=df_limited.select(['veiculo_id', 'data', 'km_dia_clean']),
            target='km_dia_clean',
            min_days=self.ingestion_config.formatter.min_days,
            min_weeks=self.ingestion_config.formatter.min_weeks,
            max_gap=self.ingestion_config.formatter.max_gap
        )
        
        formatter_h = DatasetFormatter(
            df=df_limited.select(['veiculo_id', 'data', 'h_dia_clean']),
            target='h_dia_clean',
            min_days=self.ingestion_config.formatter.min_days,
            min_weeks=self.ingestion_config.formatter.min_weeks,
            max_gap=self.ingestion_config.formatter.max_gap
        )
        
        # Formatar
        df_km_formatted = formatter_km.format(output_dir=None, verbose=False)
        df_h_formatted = formatter_h.format(output_dir=None, verbose=False)
        
        if len(df_km_formatted) == 0 or len(df_h_formatted) == 0:
            logger.info(f"  ⚠️  Veículos novos com dados insuficientes para classificação")
            return pl.DataFrame(), pl.DataFrame()
        
        # Obter conjuntos de veículos
        if isinstance(df_km_formatted, pl.DataFrame):
            km_vehicle_ids = set(df_km_formatted['veiculo_id'].unique().to_list())
            h_vehicle_ids = set(df_h_formatted['veiculo_id'].unique().to_list())
        else:
            km_vehicle_ids = set(df_km_formatted['veiculo_id'].unique())
            h_vehicle_ids = set(df_h_formatted['veiculo_id'].unique())
        
        # 🔍 VERIFICAÇÃO PRÉ-CLASSIFICAÇÃO
        logger.info(f"  🔍 Verificação pré-classificação:")
        logger.info(f"     Veículos em df_km_formatted: {len(km_vehicle_ids)}")
        logger.info(f"     Veículos em df_h_formatted: {len(h_vehicle_ids)}")
        
        # Veículos que passaram apenas em uma das formatações
        only_km = km_vehicle_ids - h_vehicle_ids
        only_h = h_vehicle_ids - km_vehicle_ids
        both = km_vehicle_ids & h_vehicle_ids
        
        logger.info(f"     Apenas KM (outliers - serão excluídos): {len(only_km)}")
        logger.info(f"     Apenas H (classificação direta): {len(only_h)}")
        logger.info(f"     Ambos (requerem classificação): {len(both)}")
        
        # Inicializar listas de resultados
        km_ids = []
        h_ids = []
        
        # 1. Veículos apenas em H → atribuir diretamente como H
        if only_h:
            h_ids.extend(list(only_h))
            logger.info(f"  ✅ {len(only_h)} veículos classificados diretamente como H")
            if len(only_h) <= 5:
                logger.info(f"     IDs: {list(only_h)}")
        
        # 2. Veículos apenas em KM → EXCLUIR (outliers)
        if only_km:
            logger.warning(f"  ⚠️  {len(only_km)} veículos excluídos (outliers - apenas KM válido)")
            if len(only_km) <= 5:
                logger.warning(f"     IDs excluídos: {list(only_km)}")
        
        # 3. Veículos em ambos → CLASSIFICAR com modelo
        if both:
            logger.info(f"  🤖 Classificando {len(both)} veículos com modelo...")
            
            # Merge apenas dos veículos que estão em ambos
            merger = DatasetMerger(verbose=False)
            df_merged = merger.merge_from_dataframes(
                df_raw=df_limited.filter(pl.col('veiculo_id').is_in(list(both))),
                formatted_dfs={
                    'km_dia_clean': df_km_formatted.filter(pl.col('veiculo_id').is_in(list(both))) if isinstance(df_km_formatted, pl.DataFrame) else df_km_formatted[df_km_formatted['veiculo_id'].isin(both)],
                    'h_dia_clean': df_h_formatted.filter(pl.col('veiculo_id').is_in(list(both))) if isinstance(df_h_formatted, pl.DataFrame) else df_h_formatted[df_h_formatted['veiculo_id'].isin(both)]
                }
            )
            
            # Carregar modelo de classificação
            predictor_type = VehiclePredictor(
                model_path=self.config.classification.models.stage1_type
            )
            
            # Classificar
            result = predictor_type.predict_batch(df_merged)[['veiculo_id', 'prediction']]
            result['prediction'] = result['prediction'].map(self.config.mapping.category)
            
            # Adicionar aos resultados
            km_ids_model = result.loc[result['prediction'] == 'km', 'veiculo_id'].to_list()
            h_ids_model = result.loc[result['prediction'] == 'h', 'veiculo_id'].to_list()
            
            km_ids.extend(km_ids_model)
            h_ids.extend(h_ids_model)
            
            logger.info(f"     {len(km_ids_model)} classificados como KM")
            logger.info(f"     {len(h_ids_model)} classificados como H")
        
        # Resumo final
        logger.info(f"  📊 Resultado final da classificação:")
        logger.info(f"     Total KM: {len(km_ids)}")
        logger.info(f"     Total H: {len(h_ids)}")
        logger.info(f"     Total excluídos: {len(only_km)}")
        
        # Filtrar dataframes finais
        df_km_new = df_limited.filter(pl.col('veiculo_id').is_in(km_ids)).select(['veiculo_id', 'data', 'km_dia_clean'])
        df_h_new = df_limited.filter(pl.col('veiculo_id').is_in(h_ids)).select(['veiculo_id', 'data', 'h_dia_clean'])
        
        return df_km_new, df_h_new

    def _concatenate(
        self,
        df_existing: pl.DataFrame,
        df_new: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Concatena dataframes
        """
        if len(df_existing) == 0 and len(df_new) == 0:
            return pl.DataFrame()
        
        if len(df_existing) == 0:
            df_combined = df_new
        elif len(df_new) == 0:
            df_combined = df_existing
        else:
            df_combined = pl.concat([df_existing, df_new])
        
        return df_combined


    async def _classify_all_segments(
        self,
        df_km: pl.DataFrame,
        df_h: pl.DataFrame
    ) -> Dict[int, int]:
        """
        Classifica segments de TODOS os veículos (existentes + novos)
        
        Returns
        -------
        dict
            {vehicle_id: segment}
        """
        logger.info("  Classificando segments...")
        
        segments = {}
        
        out_dir = Path(__file__).parent.parent
        # Classificar KM
        if len(df_km) > 0:
            predictor_km = VehiclePredictor(
                model_path=self.config.classification.models.stage2_displacement
            )
                        
            result_km = predictor_km.predict_batch(df_km)[['veiculo_id', 'prediction']]
            segments.update(dict(result_km.itertuples(index=False)))

        # Classificar H
        if len(df_h) > 0:
            predictor_h = VehiclePredictor(
                model_path=self.config.classification.models.stage2_machine
            )
            
            result_h = predictor_h.predict_batch(df_h)[['veiculo_id', 'prediction']]
            segments.update(dict(result_h.itertuples(index=False)))

        logger.info(f"  ✅ {len(segments)} segments classificados")        
        
        return segments


    async def _extract_and_save(
        self,
        df: pl.DataFrame,
        category: str,
        segments: Dict[int, int]
    ) -> int:
        """
        Extrai features e salva no banco
        
        Returns
        -------
        int
            Número de veículos salvos
        """
        if len(df) == 0:
            return 0
        
        logger.info(f"  Extraindo features {category.upper()}...")
        
        # Extrair features
        target = f'{category}_dia_clean'
        
        seg_extractor = SegmentationFeatureExtractor(metric=category)
        df_summary = seg_extractor.extract(df)
        
        wd_extractor = WeekdayFeatureExtractor(metric=category)
        df_weekday = wd_extractor.extract(df)
        
        # Preparar dados para salvar
        vehicles_data = []
        summary_data = []
        weekday_data = []
        
        for vid in df['veiculo_id'].unique().to_list():
            df_vehicle = df.filter(pl.col('veiculo_id') == vid).sort('data')
            
            samples = df_vehicle[target].to_list()
            samples_start_date = df_vehicle['data'].min()
            samples_end_date = df_vehicle['data'].max()
            
            # Upper
            p_upper = self.config.profile.p_upper
            upper = np.percentile(samples, p_upper)
            
            vehicles_data.append({
                'id': vid,
                'category': VehicleCategory(category.lower()),
                'segment': segments[vid],
                'samples_start_date': samples_start_date,
                'samples_end_date': samples_end_date,
                'upper': float(upper),
                'samples': samples
            })
            
            # Summary
            row_summary = df_summary[df_summary['veiculo_id'] == vid].iloc[0]
            
            summary_data.append({
                'vehicle_id': vid,
                'per_day': float(row_summary[f'{category}_por_dia']),
                'mean': float(row_summary[f'media_{category}']),
                'max': float(row_summary[f'max_{category}']),
                'median': float(row_summary[f'mediana_{category}']),
                'std': float(row_summary[f'std_{category}']),
                'continuity_score': float(row_summary[f'score_continuidade_{category}']),
                'active_weeks_rate': float(row_summary[f'taxa_semanas_ativas_{category}']),
                'active_days_rate': float(row_summary[f'taxa_dias_ativos_{category}']),
                'gaps_cv': float(row_summary[f'cv_gaps_{category}']),
                'mean_gap': float(row_summary[f'gap_medio_{category}']),
                'max_gap': float(row_summary[f'gap_max_{category}']),
                'cv': float(row_summary[f'cv_{category}']),
                'p25': float(row_summary[f'p25_{category}']),
                'p75': float(row_summary[f'p75_{category}']),
                'iqr': float(row_summary[f'iqr_{category}'])
            })
            
            # Weekday
            row_weekday = df_weekday[df_weekday['veiculo_id'] == vid].iloc[0]
            
            for day in range(1, 8):
                weekday_data.append({
                    'vehicle_id': vid,
                    'day': day,
                    'mean': float(row_weekday[f'day_{day}_mean']),
                    'std': float(row_weekday[f'day_{day}_std']),
                    'median': float(row_weekday[f'day_{day}_median']),
                    'max': float(row_weekday[f'day_{day}_max']),
                    'min': float(row_weekday[f'day_{day}_min']),
                    'p25': float(row_weekday[f'day_{day}_p25']),
                    'p75': float(row_weekday[f'day_{day}_p75']),
                    'iqr': float(row_weekday[f'day_{day}_iqr']),
                    'prob_active': float(row_weekday[f'day_{day}_prob_active']),
                    'cv': float(row_weekday[f'day_{day}_cv'])
                })
        
        # Salvar
        await self._upsert_vehicles_batch(vehicles_data)
        await self._upsert_summary_features_batch(summary_data)
        await self._upsert_weekday_features_batch(weekday_data)
        
        logger.info(f"  ✅ {len(vehicles_data)} veículos {category.upper()} salvos")
        
        return len(vehicles_data)
    
    @staticmethod
    def _get_category_from_target(target: str) -> VehicleCategory:
        """Extrai categoria do target"""
        metric = target.split('_')[0]
        if metric == 'km':
            return VehicleCategory.KM
        elif metric == 'h':
            return VehicleCategory.H
        else:
            raise ValueError(f"Target inválido: {target}")
    
    @staticmethod
    def _validate_and_parse_segmentation(
        df_segmentation: pd.DataFrame,
        target: str
    ) -> Dict[int, int]:
        """
        Valida e processa arquivo de segmentação
        Returns
        -------
        dict
            {vehicle_id: segment}
        """
        df_segmentation = df_segmentation[df_segmentation['target'] == target]
        unique_targets = df_segmentation['target'].unique()
        if target not in unique_targets:
            raise ValueError(
                f"Target incompatível no arquivo de segmentação: "
                f"arquivo={','.join(unique_targets)}, esperado={target}"
            )
        segmentation_map = dict(zip(
            df_segmentation['veiculo_id'].astype(int),
            df_segmentation['segment'].astype(int)
        ))
        logger.info(f"Segmentação carregada: {len(segmentation_map)} veículos")
        return segmentation_map
    
    @staticmethod
    def format_summary_features(
        summary: VehicleSummaryFeatures,
        category: VehicleCategory
    ) -> Dict[str, float]:
        """Formata features de resumo com prefixos da categoria"""
        metric = category.value
        return {
            f'{metric}_por_dia': summary.per_day,
            f'media_{metric}': summary.mean,
            f'max_{metric}': summary.max,
            f'mediana_{metric}': summary.median,
            f'std_{metric}': summary.std,
            f'score_continuidade_{metric}': summary.continuity_score,
            f'taxa_semanas_ativas_{metric}': summary.active_weeks_rate,
            f'taxa_dias_ativos_{metric}': summary.active_days_rate,
            f'cv_gaps_{metric}': summary.gaps_cv,
            f'gap_medio_{metric}': summary.mean_gap,
            f'gap_max_{metric}': summary.max_gap,
            f'cv_{metric}': summary.cv,
            f'p25_{metric}': summary.p25,
            f'p75_{metric}': summary.p75,
            f'iqr_{metric}': summary.iqr
        }
    
    @staticmethod
    def format_weekday_features(
        weekday_features: List[VehicleWeekdayFeatures]
    ) -> Dict[str, float]:
        """Formata features de dia da semana"""
        result = {}
        for wf in weekday_features:
            result[f'day_{wf.day}_mean'] = wf.mean
            result[f'day_{wf.day}_std'] = wf.std
            result[f'day_{wf.day}_median'] = wf.median
            result[f'day_{wf.day}_max'] = wf.max
            result[f'day_{wf.day}_min'] = wf.min
            result[f'day_{wf.day}_p25'] = wf.p25
            result[f'day_{wf.day}_p75'] = wf.p75
            result[f'day_{wf.day}_iqr'] = wf.iqr
            result[f'day_{wf.day}_prob_active'] = wf.prob_active
            result[f'day_{wf.day}_cv'] = wf.cv
        return result
    
    async def get_vehicle_profile(self, vehicle_id: int) -> Optional[Dict]:
        """
        Obtém perfil completo de um veículo
        
        Parameters
        ----------
        vehicle_id : int
            ID do veículo
        
        Returns
        -------
        dict ou None
            Perfil completo ou None se não encontrado
        """
        # Buscar veículo
        stmt = select(Vehicle).where(Vehicle.id == vehicle_id)
        result = await self.session.execute(stmt)
        vehicle = result.scalar_one_or_none()
        
        if not vehicle:
            return None
        
        # Buscar summary features
        stmt = select(VehicleSummaryFeatures).where(
            VehicleSummaryFeatures.vehicle_id == vehicle_id
        )
        result = await self.session.execute(stmt)
        summary = result.scalar_one_or_none()
        
        # Buscar weekday features
        stmt = select(VehicleWeekdayFeatures).where(
            VehicleWeekdayFeatures.vehicle_id == vehicle_id
        ).order_by(VehicleWeekdayFeatures.day)
        result = await self.session.execute(stmt)
        weekday_features = result.scalars().all()
        
        if not summary or not weekday_features:
            return None
        
        # Formatar features
        summary_dict = self.format_summary_features(summary, vehicle.category)
        weekday_dict = self.format_weekday_features(weekday_features)
        
        return {
            'vehicle_id': vehicle.id,
            'category': vehicle.category.value,
            'segment': vehicle.segment,
            'samples_start_date': vehicle.samples_start_date.isoformat(),  
            'samples_end_date': vehicle.samples_end_date.isoformat(),    
            'upper': vehicle.upper,
            'samples': vehicle.samples,
            **summary_dict,
            **weekday_dict
        }
    
    async def get_profiles_batch(self, vehicle_ids: List[int]) -> Dict[int, Dict]:
        """
        Obtém perfis de múltiplos veículos
        
        Parameters
        ----------
        vehicle_ids : List[int]
            Lista de IDs de veículos
        
        Returns
        -------
        dict
            {vehicle_id: profile_dict}
        """
        # Buscar veículos
        stmt = select(Vehicle).where(Vehicle.id.in_(vehicle_ids))
        result = await self.session.execute(stmt)
        vehicles = {v.id: v for v in result.scalars().all()}
        
        # Buscar summary features
        stmt = select(VehicleSummaryFeatures).where(
            VehicleSummaryFeatures.vehicle_id.in_(vehicle_ids)
        )
        result = await self.session.execute(stmt)
        summaries = {s.vehicle_id: s for s in result.scalars().all()}
        
        # Buscar weekday features
        stmt = select(VehicleWeekdayFeatures).where(
            VehicleWeekdayFeatures.vehicle_id.in_(vehicle_ids)
        ).order_by(VehicleWeekdayFeatures.vehicle_id, VehicleWeekdayFeatures.day)
        result = await self.session.execute(stmt)
        
        # Agrupar por vehicle_id
        weekday_by_vehicle = {}
        for wf in result.scalars().all():
            if wf.vehicle_id not in weekday_by_vehicle:
                weekday_by_vehicle[wf.vehicle_id] = []
            weekday_by_vehicle[wf.vehicle_id].append(wf)
        
        # Construir perfis
        profiles = {}
        
        for vid in vehicle_ids:
            if vid not in vehicles or vid not in summaries or vid not in weekday_by_vehicle:
                continue
            
            vehicle = vehicles[vid]
            summary = summaries[vid]
            weekday_features = weekday_by_vehicle[vid]
            
            summary_dict = self.format_summary_features(summary, vehicle.category)
            weekday_dict = self.format_weekday_features(weekday_features)
            
            profiles[vid] = {
                'vehicle_id': vehicle.id,
                'category': vehicle.category.value,
                'segment': vehicle.segment,
                'samples_start_date': vehicle.samples_start_date.isoformat(),  
                'samples_end_date': vehicle.samples_end_date.isoformat(),
                'upper': vehicle.upper,
                'samples': vehicle.samples,
                **summary_dict,
                **weekday_dict
            }
        
        return profiles
    
    async def get_vehicle_info(self, vehicle_id: int) -> Optional[Dict]:
        """
        Obtém informações básicas do veículo
        
        Parameters
        ----------
        vehicle_id : int
            ID do veículo
        
        Returns
        -------
        dict ou None
            Informações básicas ou None se não encontrado
        """
        stmt = select(Vehicle).where(Vehicle.id == vehicle_id)
        result = await self.session.execute(stmt)
        vehicle = result.scalar_one_or_none()
        
        if not vehicle:
            return None
        
        return {
            'id': vehicle.id,
            'category': vehicle.category.value,
            'segment': vehicle.segment,
            'samples_start_date': vehicle.samples_start_date.isoformat(),
            'samples_end_date': vehicle.samples_end_date.isoformat(),
            'upper': vehicle.upper,
            'n_samples': len(vehicle.samples)
        }
    
    async def get_statistics(self) -> Dict[str, Dict]:
        """
        Obtém estatísticas agregadas por categoria
        
        Returns
        -------
        dict
            Estatísticas com n_vehicles e last_update por categoria
        """
        stats = {}
        
        for category in VehicleCategory:
            # Contar veículos
            stmt = select(func.count(Vehicle.id)).where(Vehicle.category == category)
            result = await self.session.execute(stmt)
            n_vehicles = result.scalar()
            
            # Última atualização
            stmt = select(func.max(Vehicle.updated_at)).where(Vehicle.category == category)
            result = await self.session.execute(stmt)
            last_update = result.scalar()
            
            stats[category.value] = {
                'n_vehicles': n_vehicles,
                'last_update': last_update.isoformat() if last_update else None
            }
        
        return stats
    
    async def _upsert_vehicles_batch(self, vehicles_data: List[Dict]):
        """Insere ou atualiza veículos em lote (com chunking para SQLite)"""
        
        dialect_name = self.session.bind.dialect.name
        
        # SQLite: limite de variáveis, fazer em chunks
        if dialect_name == 'sqlite':
            # SQLite limite: ~999 variáveis
            # Vehicle tem 7 colunas (id, category, segment, first_date, last_date, upper, samples)
            # Chunk size = 999 / 7 ≈ 140
            chunk_size = 100
            
            for i in range(0, len(vehicles_data), chunk_size):
                chunk = vehicles_data[i:i+chunk_size]
                
                stmt = sqlite_insert(Vehicle).values(chunk)
                stmt = stmt.on_conflict_do_update(
                    index_elements=['id'],
                    set_={
                        'category': stmt.excluded.category,
                        'segment': stmt.excluded.segment,
                        'samples_start_date': stmt.excluded.samples_start_date,
                        'samples_end_date': stmt.excluded.samples_end_date,
                        'upper': stmt.excluded.upper,
                        'samples': stmt.excluded.samples,
                        'updated_at': func.now()
                    }
                )
                await self.session.execute(stmt)
            
            await self.session.commit()
            logger.info(f"Veículos salvos: {len(vehicles_data)} em {len(range(0, len(vehicles_data), chunk_size))} chunks")
        
        elif dialect_name == 'postgresql':
            # PostgreSQL não tem esse limite
            stmt = insert(Vehicle).values(vehicles_data)
            stmt = stmt.on_conflict_do_update(
                index_elements=['id'],
                set_={
                    'category': stmt.excluded.category,
                    'segment': stmt.excluded.segment,
                    'samples_start_date': stmt.excluded.samples_start_date,
                    'samples_end_date': stmt.excluded.samples_end_date,
                    'upper': stmt.excluded.upper,
                    'samples': stmt.excluded.samples,
                    'updated_at': func.now()
                }
            )
            await self.session.execute(stmt)
            await self.session.commit()
        
        else:
            # Fallback: um por vez
            for record in vehicles_data:
                stmt = select(Vehicle).where(Vehicle.id == record['id'])
                result = await self.session.execute(stmt)
                existing = result.scalar_one_or_none()
                
                if existing:
                    for key, value in record.items():
                        if key != 'id':
                            setattr(existing, key, value)
                    existing.updated_at = datetime.utcnow()
                else:
                    new_vehicle = Vehicle(**record)
                    self.session.add(new_vehicle)
            
            await self.session.commit()


    async def _upsert_summary_features_batch(self, summary_data: List[Dict]):
        """Insere ou atualiza summary features em lote (com chunking)"""

        dialect_name = self.session.bind.dialect.name
        
        update_cols = [
            'per_day', 'mean', 'max', 'median', 'std',
            'continuity_score', 'active_weeks_rate', 'active_days_rate',
            'gaps_cv', 'mean_gap', 'max_gap', 'cv', 'p25', 'p75', 'iqr'
        ]
        
        if dialect_name == 'sqlite':
            # Summary tem 16 colunas (vehicle_id + 15 features)
            # Chunk size = 999 / 16 ≈ 60
            chunk_size = 60
            
            for i in range(0, len(summary_data), chunk_size):
                chunk = summary_data[i:i+chunk_size]
                
                stmt = sqlite_insert(VehicleSummaryFeatures).values(chunk)
                update_dict = {col: getattr(stmt.excluded, col) for col in update_cols}
                update_dict['updated_at'] = func.now()
                stmt = stmt.on_conflict_do_update(
                    index_elements=['vehicle_id'],
                    set_=update_dict
                )
                await self.session.execute(stmt)
            
            await self.session.commit()
            logger.info(f"Summary features salvos: {len(summary_data)} em {len(range(0, len(summary_data), chunk_size))} chunks")
        
        elif dialect_name == 'postgresql':
            stmt = insert(VehicleSummaryFeatures).values(summary_data)
            update_dict = {col: getattr(stmt.excluded, col) for col in update_cols}
            update_dict['updated_at'] = func.now()
            stmt = stmt.on_conflict_do_update(
                index_elements=['vehicle_id'],
                set_=update_dict
            )
            await self.session.execute(stmt)
            await self.session.commit()
        
        else:
            for record in summary_data:
                stmt = select(VehicleSummaryFeatures).where(
                    VehicleSummaryFeatures.vehicle_id == record['vehicle_id']
                )
                result = await self.session.execute(stmt)
                existing = result.scalar_one_or_none()
                
                if existing:
                    for key, value in record.items():
                        if key != 'vehicle_id':
                            setattr(existing, key, value)
                    existing.updated_at = datetime.utcnow()
                else:
                    new_summary = VehicleSummaryFeatures(**record)
                    self.session.add(new_summary)
            
            await self.session.commit()


    async def _upsert_weekday_features_batch(self, weekday_data: List[Dict]):
        """Insere ou atualiza weekday features em lote (com chunking)"""
        dialect_name = self.session.bind.dialect.name
        
        update_cols = [
            'mean', 'std', 'median', 'max', 'min',
            'p25', 'p75', 'iqr', 'prob_active', 'cv'
        ]
        
        if dialect_name == 'sqlite':
            # Weekday tem 12 colunas (vehicle_id, day + 10 features)
            # Chunk size = 999 / 12 ≈ 80
            chunk_size = 80
            
            for i in range(0, len(weekday_data), chunk_size):
                chunk = weekday_data[i:i+chunk_size]
                
                stmt = sqlite_insert(VehicleWeekdayFeatures).values(chunk)
                update_dict = {col: getattr(stmt.excluded, col) for col in update_cols}
                update_dict['updated_at'] = func.now()
                stmt = stmt.on_conflict_do_update(
                    index_elements=['vehicle_id', 'day'],
                    set_=update_dict
                )
                await self.session.execute(stmt)
            
            await self.session.commit()
            logger.info(f"Weekday features salvos: {len(weekday_data)} em {len(range(0, len(weekday_data), chunk_size))} chunks")
        
        elif dialect_name == 'postgresql':
            stmt = insert(VehicleWeekdayFeatures).values(weekday_data)
            update_dict = {col: getattr(stmt.excluded, col) for col in update_cols}
            update_dict['updated_at'] = func.now()
            stmt = stmt.on_conflict_do_update(
                index_elements=['vehicle_id', 'day'],
                set_=update_dict
            )
            await self.session.execute(stmt)
            await self.session.commit()
        
        else:
            for record in weekday_data:
                stmt = select(VehicleWeekdayFeatures).where(
                    and_(
                        VehicleWeekdayFeatures.vehicle_id == record['vehicle_id'],
                        VehicleWeekdayFeatures.day == record['day']
                    )
                )
                result = await self.session.execute(stmt)
                existing = result.scalar_one_or_none()
                
                if existing:
                    for key, value in record.items():
                        if key not in ['vehicle_id', 'day']:
                            setattr(existing, key, value)
                    existing.updated_at = datetime.utcnow()
                else:
                    new_weekday = VehicleWeekdayFeatures(**record)
                    self.session.add(new_weekday)
            
            await self.session.commit()

    async def get_all_vehicles_category(self) -> pd.DataFrame:
        """
        Retorna category e segment de todos os veículos
        
        Returns
        -------
        pd.DataFrame
            DataFrame com colunas: veiculo_id, category, segment
        
        Examples
        --------
        >>> df = await service.get_all_vehicles_category()
        >>> print(df.head())
        veiculo_id category  segment
        0        1316       km        2
        1       18230       km        0
        2       10495        h        1
        """
        stmt = select(
            Vehicle.id,
            Vehicle.category,
            Vehicle.segment
        ).order_by(Vehicle.id)
        
        result = await self.session.execute(stmt)
        rows = result.all()
        
        # Converter para DataFrame
        df = pd.DataFrame(
            [(row.id, row.category.value, row.segment) for row in rows],
            columns=['veiculo_id', 'category', 'segment']
        )
        
        logger.info(f"Retornados {len(df)} veículos")
        
        return df


    async def get_vehicles_by_category(self, category: str) -> pd.DataFrame:
        """
        Retorna veículos de uma categoria específica
        
        Parameters
        ----------
        category : str
            Categoria ('km' ou 'h')
        
        Returns
        -------
        pd.DataFrame
            DataFrame com colunas: veiculo_id, category, segment
        
        Examples
        --------
        >>> df = await service.get_vehicles_by_category('km')
        >>> print(f"Veículos KM: {len(df)}")
        >>> print(df['segment'].value_counts())
        """
        # Validar categoria
        try:
            cat_enum = VehicleCategory(category)
        except ValueError:
            raise ValueError(f"Categoria inválida: {category}. Use 'km' ou 'h'")
        
        stmt = select(
            Vehicle.id,
            Vehicle.category,
            Vehicle.segment
        ).where(
            Vehicle.category == cat_enum
        ).order_by(Vehicle.id)
        
        result = await self.session.execute(stmt)
        rows = result.all()
        
        df = pd.DataFrame(
            [(row.id, row.category.value, row.segment) for row in rows],
            columns=['veiculo_id', 'category', 'segment']
        )
        
        logger.info(f"Retornados {len(df)} veículos da categoria {category}")
        
        return df


    async def get_vehicles_by_segment(
        self,
        category: str,
        segment: int
    ) -> pd.DataFrame:
        """
        Retorna veículos de uma categoria e segmento específicos
        
        Parameters
        ----------
        category : str
            Categoria ('km' ou 'h')
        segment : int
            Número do segmento (0, 1, 2, ...)
        
        Returns
        -------
        pd.DataFrame
            DataFrame com colunas: veiculo_id, category, segment
        
        Examples
        --------
        >>> df = await service.get_vehicles_by_segment('km', 2)
        >>> print(f"Veículos KM segmento 2: {len(df)}")
        """
        # Validar categoria
        try:
            cat_enum = VehicleCategory(category)
        except ValueError:
            raise ValueError(f"Categoria inválida: {category}. Use 'km' ou 'h'")
        
        stmt = select(
            Vehicle.id,
            Vehicle.category,
            Vehicle.segment
        ).where(
            Vehicle.category == cat_enum,
            Vehicle.segment == segment
        ).order_by(Vehicle.id)
        
        result = await self.session.execute(stmt)
        rows = result.all()
        
        df = pd.DataFrame(
            [(row.id, row.category.value, row.segment) for row in rows],
            columns=['veiculo_id', 'category', 'segment']
        )
        
        logger.info(
            f"Retornados {len(df)} veículos da categoria {category} segmento {segment}"
        )
        
        return df

    async def _delete_vehicles(self, vehicle_ids: List[int]):
        """
        Exclui veículos do banco de dados
        
        Como Vehicle tem CASCADE, isso automaticamente remove:
        - VehicleSummaryFeatures
        - VehicleWeekdayFeatures
        
        Parameters
        ----------
        vehicle_ids : List[int]
            IDs dos veículos a excluir
        """
        if not vehicle_ids:
            return
        
        from sqlalchemy import delete
        
        # Excluir veículos (cascade remove features automaticamente)
        stmt = delete(Vehicle).where(Vehicle.id.in_(vehicle_ids))
        await self.session.execute(stmt)
        await self.session.commit()
        
        logger.info(f"  ✅ {len(vehicle_ids)} veículos excluídos do banco")

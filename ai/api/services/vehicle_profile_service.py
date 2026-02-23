# api/services/vehicle_profile_service.py

from typing import List, Dict, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from datetime import datetime
from pathlib import Path
import pandas as pd
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

logger = logging.getLogger(__name__)


class VehicleProfileService:
    """Service para gerenciar perfis de veículos"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
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
    
    async def import_from_file(
        self,
        target: str,
        profile_dir: str,
        df_segmentation: pd.DataFrame
    ) -> Dict[str, int]:
        """Importa perfis de arquivos do VehicleProfile"""
        logger.info(f"Importando perfis de {profile_dir} para target={target}")
        
        segmentation_map = self._validate_and_parse_segmentation(df_segmentation, target)
        
        vp = VehicleProfile.load(profile_dir)
        
        if vp.target != target:
            raise ValueError(f"Target incompatível: arquivo={vp.target}, solicitado={target}")
        
        category = self._get_category_from_target(target)
        metric = vp.metric
        
        df_latest = (
            vp.df_versions
            .sort_values(['veiculo_id', 'year', 'week'])
            .groupby('veiculo_id')
            .last()
            .reset_index()
        )
        
        df_period_latest = (
            vp.df_effective_period
            .sort_values(['veiculo_id', 'year', 'week'])
            .groupby('veiculo_id')
            .last()
            .reset_index()
        )
        
        vehicles_data = []
        summary_data = []
        weekday_data = []
        vehicles_without_segment = []
        
        for _, row in df_latest.iterrows():
            vehicle_id = int(row['veiculo_id'])
            
            if vehicle_id not in segmentation_map:
                vehicles_without_segment.append(vehicle_id)
                logger.warning(f"Veículo {vehicle_id} não encontrado no arquivo de segmentação")
                continue
            
            segment = segmentation_map[vehicle_id]
            period_row = df_period_latest[df_period_latest['veiculo_id'] == vehicle_id].iloc[0]
            samples = vp.target_samples.get(vehicle_id, [])
            
            vehicles_data.append({
                'id': vehicle_id,
                'category': category,
                'segment': segment,
                'first_activity_date': period_row['dt_inicio'].date(),
                'last_activity_date': period_row['dt_fim'].date(),
                'upper': float(row['upper']),
                'samples': samples
            })
            
            summary_data.append({
                'vehicle_id': vehicle_id,
                'per_day': float(row[f'{metric}_por_dia']),
                'mean': float(row[f'media_{metric}']),
                'max': float(row[f'max_{metric}']),
                'median': float(row[f'mediana_{metric}']),
                'std': float(row[f'std_{metric}']),
                'continuity_score': float(row[f'score_continuidade_{metric}']),
                'active_weeks_rate': float(row[f'taxa_semanas_ativas_{metric}']),
                'active_days_rate': float(row[f'taxa_dias_ativos_{metric}']),
                'gaps_cv': float(row[f'cv_gaps_{metric}']),
                'mean_gap': float(row[f'gap_medio_{metric}']),
                'max_gap': float(row[f'gap_max_{metric}']),
                'cv': float(row[f'cv_{metric}']),
                'p25': float(row[f'p25_{metric}']),
                'p75': float(row[f'p75_{metric}']),
                'iqr': float(row[f'iqr_{metric}'])
            })
            
            for day in range(1, 8):
                weekday_data.append({
                    'vehicle_id': vehicle_id,
                    'day': day,
                    'mean': float(row[f'day_{day}_mean']),
                    'std': float(row[f'day_{day}_std']),
                    'median': float(row[f'day_{day}_median']),
                    'max': float(row[f'day_{day}_max']),
                    'min': float(row[f'day_{day}_min']),
                    'p25': float(row[f'day_{day}_p25']),
                    'p75': float(row[f'day_{day}_p75']),
                    'iqr': float(row[f'day_{day}_iqr']),
                    'prob_active': float(row[f'day_{day}_prob_active']),
                    'cv': float(row[f'day_{day}_cv'])
                })
        
        if vehicles_without_segment:
            logger.warning(
                f"{len(vehicles_without_segment)} veículos sem segmento foram ignorados"
            )
        
        await self._upsert_vehicles_batch(vehicles_data)
        await self._upsert_summary_features_batch(summary_data)
        await self._upsert_weekday_features_batch(weekday_data)
        
        logger.info(f"Importação concluída: {len(vehicles_data)} veículos")
        
        return {
            'target': target,
            'category': category.value,
            'n_vehicles': len(vehicles_data)
        }
    
    async def update_from_dataframe(
        self,
        target: str,
        df: pd.DataFrame,
        vehicle_ids: Optional[List[int]] = None
    ) -> Dict[str, int]:
        """
        Atualiza perfis a partir de DataFrame
        
        Fluxo:
        1. Recupera veículos existentes e seus samples
        2. Atualiza samples com novos dados
        3. Recalcula métricas usando extractors
        
        Parameters
        ----------
        target : str
            Nome do target
        df : pd.DataFrame
            DataFrame com colunas: veiculo_id, data, {target}
        vehicle_ids : List[int], optional
            Se fornecido, atualiza apenas esses veículos
        
        Returns
        -------
        dict
            Estatísticas da atualização
        """
        logger.info(f"Atualizando perfis para target={target}")
        
        category = self._get_category_from_target(target)
        metric = category.value
        
        # 1. Recuperar veículos existentes
        unique_vehicles = df['veiculo_id'].unique().tolist()
        
        if vehicle_ids:
            unique_vehicles = [v for v in unique_vehicles if v in vehicle_ids]
        
        stmt = select(Vehicle).where(
            Vehicle.id.in_(unique_vehicles),
            Vehicle.category == category
        )
        result = await self.session.execute(stmt)
        existing_vehicles = {v.id: v for v in result.scalars().all()}
        
        logger.info(f"Recuperados {len(existing_vehicles)} veículos existentes")
        
        # 2. Atualizar samples e reconstruir séries
        sample_size = 364
        p_upper = 99
        
        updated_vehicles_data = []
        df_for_features = []
        
        for vehicle_id, vehicle in existing_vehicles.items():
            df_vehicle = df[df['veiculo_id'] == vehicle_id].copy()
            
            if df_vehicle.empty:
                continue
            
            # Obter samples existentes
            old_samples = vehicle.samples
            
            # Obter novos valores
            new_values = df_vehicle[target].dropna().tolist()
            
            # Atualizar samples (últimos sample_size valores)
            combined_samples = old_samples + new_values
            updated_samples = combined_samples[-sample_size:]
            
            # Calcular novo período efetivo
            df_vehicle_active = df_vehicle[df_vehicle[target] > 0]
            
            if not df_vehicle_active.empty:
                new_first_date = df_vehicle_active['data'].min()
                new_last_date = df_vehicle_active['data'].max()
                
                first_date = min(vehicle.first_activity_date, new_first_date.date())
                last_date = max(vehicle.last_activity_date, new_last_date.date())
            else:
                first_date = vehicle.first_activity_date
                last_date = vehicle.last_activity_date
            
            # Calcular novo upper
            new_upper = np.percentile(updated_samples, p_upper) if len(updated_samples) > 0 else vehicle.upper
            
            updated_vehicles_data.append({
                'id': vehicle_id,
                'category': category,
                'segment': vehicle.segment,
                'first_activity_date': first_date,
                'last_activity_date': last_date,
                'upper': float(new_upper),
                'samples': updated_samples
            })
            
            # Reconstruir série para extração de features
            date_range = pd.date_range(start=first_date, end=last_date, freq='D')
            n_samples = len(updated_samples)
            
            if n_samples > 0:
                df_vehicle_series = pd.DataFrame({
                    'veiculo_id': vehicle_id,
                    'data': date_range[-n_samples:],
                    target: updated_samples
                })
                df_for_features.append(df_vehicle_series)
        
        if not df_for_features:
            logger.warning("Nenhum veículo para atualizar")
            return {'target': target, 'n_vehicles': 0}
        
        # 3. Recalcular métricas
        df_combined = pd.concat(df_for_features, ignore_index=True)
        
        # Extrair summary features
        seg_extractor = SegmentationFeatureExtractor(metric=metric)
        df_summary = seg_extractor.extract(df_combined)
        
        # Extrair weekday features
        wd_extractor = WeekdayFeatureExtractor(metric=metric)
        df_weekday = wd_extractor.extract(df_combined)
        
        # Preparar dados
        summary_data = []
        weekday_data = []
        
        for _, row in df_summary.iterrows():
            vehicle_id = int(row['veiculo_id'])
            
            summary_data.append({
                'vehicle_id': vehicle_id,
                'per_day': float(row[f'{metric}_por_dia']),
                'mean': float(row[f'media_{metric}']),
                'max': float(row[f'max_{metric}']),
                'median': float(row[f'mediana_{metric}']),
                'std': float(row[f'std_{metric}']),
                'continuity_score': float(row[f'score_continuidade_{metric}']),
                'active_weeks_rate': float(row[f'taxa_semanas_ativas_{metric}']),
                'active_days_rate': float(row[f'taxa_dias_ativos_{metric}']),
                'gaps_cv': float(row[f'cv_gaps_{metric}']),
                'mean_gap': float(row[f'gap_medio_{metric}']),
                'max_gap': float(row[f'gap_max_{metric}']),
                'cv': float(row[f'cv_{metric}']),
                'p25': float(row[f'p25_{metric}']),
                'p75': float(row[f'p75_{metric}']),
                'iqr': float(row[f'iqr_{metric}'])
            })
        
        for _, row in df_weekday.iterrows():
            vehicle_id = int(row['veiculo_id'])
            
            for day in range(1, 8):
                weekday_data.append({
                    'vehicle_id': vehicle_id,
                    'day': day,
                    'mean': float(row[f'day_{day}_mean']),
                    'std': float(row[f'day_{day}_std']),
                    'median': float(row[f'day_{day}_median']),
                    'max': float(row[f'day_{day}_max']),
                    'min': float(row[f'day_{day}_min']),
                    'p25': float(row[f'day_{day}_p25']),
                    'p75': float(row[f'day_{day}_p75']),
                    'iqr': float(row[f'day_{day}_iqr']),
                    'prob_active': float(row[f'day_{day}_prob_active']),
                    'cv': float(row[f'day_{day}_cv'])
                })
        
        # Salvar
        await self._upsert_vehicles_batch(updated_vehicles_data)
        await self._upsert_summary_features_batch(summary_data)
        await self._upsert_weekday_features_batch(weekday_data)
        
        logger.info(f"Atualização concluída: {len(updated_vehicles_data)} veículos")
        
        return {
            'target': target,
            'n_vehicles': len(updated_vehicles_data)
        }

    async def update_from_csv_path(
        self,
        target: str,
        csv_path: str
    ) -> Dict[str, int]:
        """Atualiza perfis lendo CSV diretamente do filesystem compartilhado."""
        csv_file = Path(csv_path)
        if not csv_file.exists():
            raise FileNotFoundError(f'Arquivo CSV não encontrado: {csv_path}')

        df = pd.read_csv(csv_file)
        required_cols = ['veiculo_id', target]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f'Colunas faltando no CSV: {missing}')

        if 'data_br' in df.columns and df['data_br'].notna().any():
            dt = pd.to_datetime(df['data_br'], format='%d/%m/%Y', errors='coerce')
        elif 'data' in df.columns:
            dt = pd.to_datetime(df['data'], errors='coerce')
            if dt.isna().all():
                dt = pd.to_datetime(df['data'], dayfirst=True, errors='coerce')
        else:
            raise ValueError("CSV sem coluna de data ('data' ou 'data_br').")

        normalized_df = pd.DataFrame({
            'veiculo_id': pd.to_numeric(df['veiculo_id'], errors='coerce'),
            'data': dt,
            target: pd.to_numeric(df[target], errors='coerce'),
        })
        normalized_df = normalized_df.dropna(subset=['veiculo_id', 'data', target])
        if normalized_df.empty:
            logger.warning('CSV sem linhas válidas para target=%s', target)
            return {'target': target, 'n_vehicles': 0}

        normalized_df['veiculo_id'] = normalized_df['veiculo_id'].astype('int64')
        normalized_df = normalized_df.drop_duplicates(subset=['veiculo_id', 'data'], keep='last')

        return await self.update_from_dataframe(
            target=target,
            df=normalized_df[['veiculo_id', 'data', target]],
        )
    
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
            'first_activity_date': vehicle.first_activity_date.isoformat(),  
            'last_activity_date': vehicle.last_activity_date.isoformat(),    
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
                'first_activity_date': vehicle.first_activity_date.isoformat(),  
                'last_activity_date': vehicle.last_activity_date.isoformat(),
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
            'first_activity_date': vehicle.first_activity_date.isoformat(),
            'last_activity_date': vehicle.last_activity_date.isoformat(),
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
        from sqlalchemy.dialects.postgresql import insert
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert
        
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
                        'first_activity_date': stmt.excluded.first_activity_date,
                        'last_activity_date': stmt.excluded.last_activity_date,
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
                    'first_activity_date': stmt.excluded.first_activity_date,
                    'last_activity_date': stmt.excluded.last_activity_date,
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
        from sqlalchemy.dialects.postgresql import insert
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert
        
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
        from sqlalchemy.dialects.postgresql import insert
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert
        
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

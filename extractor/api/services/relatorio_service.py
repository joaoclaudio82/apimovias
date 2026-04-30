import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import Optional, Set, Tuple, List, Callable, Dict, Iterable
from starlette.responses import FileResponse
from api.repositories.relatorio_repository import RelatorioRepository
from api.settings import settings
from api.utils.path_utils import PathUtils
from api.utils.csv_utils import CsvUtils

class RelatorioService:
    __FEATURE_WINDOWS: List[int] = [7, 14, 21, 28]
    __OUTPUT_COLUMNS: List[str] = [
        'veiculo_id',
        'data',
        'data_br',
        'km_dia_clean',
        'h_dia_clean',
        'mm_km_7',
        'sm_km_7',
        'mm_h_7',
        'mm_km_14',
        'sm_km_14',
        'mm_h_14',
        'mm_km_21',
        'sm_km_21',
        'mm_h_21',
        'mm_km_28',
        'sm_km_28',
        'mm_h_28',
        'weekday',
        'fimdesemana',
        'mes',
        'sin_sem',
        'cos_sem',
        'flag_parado',
        'dias_parado_seq',
        'pct_parado_7d',
        'pct_parado_14d',
    ]

    @staticmethod
    def download(path: Path = settings.CSV_PATH) -> Optional[FileResponse]:
        if PathUtils.is_empty(path):
            return None
        return FileResponse(path, filename=path.name, media_type='text/csv')

    @classmethod
    def create_and_append_csv(
        cls,
        id_start: int,
        id_end: int,
        data_ini: Optional[date],
        data_fim: Optional[date],
        path: Path = settings.CSV_PATH
    ) -> None:
        cls.__validate_existing_csv_schema(path)
        existing_keys: Set[Tuple[str, str]] = cls._existing_keys(path)
        latest_by_vehicle = cls._latest_dates_by_vehicle(path)
        vei_ids: List[int] = []
        data_ini_list: List[Optional[date]] = []
        data_fim_list: List[Optional[date]] = []
        for veiculo_id in range(id_start, id_end + 1):
            latest = latest_by_vehicle.get(veiculo_id)
            if data_fim is not None and latest is not None and latest >= data_fim:
                continue

            data_ini_eff = data_ini
            if latest is not None:
                next_day = latest + timedelta(days=1)
                if data_ini_eff is None or next_day > data_ini_eff:
                    data_ini_eff = next_day

            vei_ids.append(veiculo_id)
            data_ini_list.append(data_ini_eff)
            data_fim_list.append(data_fim)

        if not vei_ids:
            return

        batch_size = max(1, int(settings.RELATORIO_QUERY_BATCH_SIZE))
        parallel_workers = max(1, int(settings.RELATORIO_QUERY_PARALLEL_WORKERS))
        param_batches = list(cls.__iter_param_batches(
            vei_ids,
            data_ini_list,
            data_fim_list,
            batch_size=batch_size
        ))

        worker_count = min(parallel_workers, len(param_batches))
        if worker_count <= 1:
            for batch_vei_ids, batch_data_ini, batch_data_fim in param_batches:
                for veiculos in RelatorioRepository.get_veiculos_stream(
                    batch_vei_ids,
                    batch_data_ini,
                    batch_data_fim,
                    vehicles_per_chunk=batch_size
                ):
                    cls.__append_vehicle_rows(veiculos, existing_keys, path)
            return

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    cls.__fetch_batch_rows,
                    batch_vei_ids,
                    batch_data_ini,
                    batch_data_fim,
                    batch_size
                )
                for batch_vei_ids, batch_data_ini, batch_data_fim in param_batches
            ]
            try:
                for future in as_completed(futures):
                    veiculos = future.result()
                    if veiculos:
                        cls.__append_vehicle_rows(veiculos, existing_keys, path)
            except Exception:
                for future in futures:
                    future.cancel()
                raise

    @staticmethod
    def __fetch_batch_rows(
        vei_ids: List[int],
        data_ini_list: List[Optional[date]],
        data_fim_list: List[Optional[date]],
        batch_size: int
    ) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for veiculos in RelatorioRepository.get_veiculos_stream(
            vei_ids,
            data_ini_list,
            data_fim_list,
            vehicles_per_chunk=batch_size
        ):
            rows.extend(veiculos)
        return rows

    @classmethod
    def create_and_append_csv_all(
        cls,
        path: Path = settings.CSV_PATH
    ) -> None:
        cls.__validate_existing_csv_schema(path)
        existing_keys: Set[Tuple[str, str]] = cls._existing_keys(path)
        batch_size = max(1, int(settings.RELATORIO_QUERY_BATCH_SIZE))
        parallel_workers = max(1, int(settings.RELATORIO_QUERY_PARALLEL_WORKERS))

        for veiculos in RelatorioRepository.get_all_veiculos_stream(
            batch_size=batch_size,
            parallel_workers=parallel_workers
        ):
            cls.__append_vehicle_rows(veiculos, existing_keys, path)

    @classmethod
    def __append_vehicle_rows(
        cls,
        veiculos: List[Dict[str, object]],
        existing_keys: Set[Tuple[str, str]],
        path: Path
    ) -> None:
        df: pd.DataFrame = pd.DataFrame.from_records(veiculos)
        if df.empty:
            return

        df = cls.__normalize_dataframe(df)
        df = df.drop_duplicates(subset=['veiculo_id', 'data'])
        df['__key'] = list(zip(df['veiculo_id'].astype(str), df['data'].astype(str)))
        df = df[~df['__key'].isin(existing_keys)].drop(columns='__key')
        if df.empty:
            return

        existing_keys.update(zip(df['veiculo_id'].astype(str), df['data'].astype(str)))
        CsvUtils.atomic_append_dataframe(df, path)

    @staticmethod
    def __iter_param_batches(
        vei_ids: List[int],
        data_ini_list: List[Optional[date]],
        data_fim_list: List[Optional[date]],
        batch_size: int
    ) -> Iterable[Tuple[List[int], List[Optional[date]], List[Optional[date]]]]:
        total = len(vei_ids)
        for start in range(0, total, batch_size):
            end = start + batch_size
            yield (
                vei_ids[start:end],
                data_ini_list[start:end],
                data_fim_list[start:end],
            )

    @staticmethod
    def _existing_keys(path: Path = settings.CSV_PATH) -> Set[Tuple[str, str]]:
        if PathUtils.is_empty(path):
            return set()

        df = pd.read_csv(path, usecols=['veiculo_id', 'data'])
        df['veiculo_id'] = df['veiculo_id'].astype(str)
        df['data'] = df['data'].astype(str)
        return set(zip(df['veiculo_id'], df['data']))

    @staticmethod
    def _latest_dates_by_vehicle(path: Path = settings.CSV_PATH) -> Dict[int, date]:
        if PathUtils.is_empty(path):
            return {}

        df = pd.read_csv(path, usecols=['veiculo_id', 'data'])
        df['data'] = pd.to_datetime(df['data'], errors='coerce')
        df = df.dropna(subset=['data'])
        if df.empty:
            return {}

        latest = df.groupby('veiculo_id')['data'].max()
        return {int(vid): dt.date() for vid, dt in latest.items()}

    @classmethod
    def __validate_existing_csv_schema(cls, path: Path = settings.CSV_PATH) -> None:
        if PathUtils.is_empty(path):
            return

        header = pd.read_csv(path, nrows=0).columns.tolist()
        if header != cls.__OUTPUT_COLUMNS:
            raise ValueError("CSV existente com colunas diferentes do esperado; remova o arquivo para gerar o novo formato.")

    @staticmethod
    def __normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        RelatorioService.__validate_schema(df)
        df['data'] = RelatorioService.__parse_date_with_fallback(df)
        dt_idx = pd.DatetimeIndex(df['data'])
        dow = pd.Series(dt_idx.weekday.tolist(), index=df.index)
        df['fimdesemana'] = dow.isin([5, 6]).astype('Int64')
        df = df.sort_values(by=['veiculo_id', 'data']).reset_index(drop=True)
        df = RelatorioService.__apply_per_vehicle(df, RelatorioService.__sane_odometer_and_select)
        df = RelatorioService.__apply_per_vehicle(df, RelatorioService.__apply_outliers)
        df = RelatorioService.__apply_per_vehicle(df, RelatorioService.__apply_features)
        df['data'] = pd.Series(pd.DatetimeIndex(df['data']).strftime('%Y-%m-%d').tolist(), index=df.index)
        df = df[RelatorioService.__OUTPUT_COLUMNS]
        return df

    @staticmethod
    def __apply_per_vehicle(df: pd.DataFrame, fn: Callable[[pd.DataFrame], pd.DataFrame]) -> pd.DataFrame:
        if 'veiculo_id' not in df.columns:
            return fn(df)

        df_idx = df.set_index('veiculo_id', drop=True)
        def _apply(g: pd.DataFrame) -> pd.DataFrame:
            if 'veiculo_id' not in g.columns:
                g = g.copy()
                g['veiculo_id'] = g.index
            return fn(g)

        res_any = df_idx.groupby(level=0, sort=False, group_keys=False).apply(_apply)
        if isinstance(res_any, pd.DataFrame):
            res: pd.DataFrame = res_any
        else:
            res = res_any.to_frame()
        if 'veiculo_id' not in res.columns:
            res = res.copy()
            res['veiculo_id'] = res.index
        return res.reset_index(drop=True)

    @staticmethod
    def __parse_date_with_fallback(df: pd.DataFrame) -> pd.Series:
        dt = pd.to_datetime(df['data'], errors='coerce')
        if dt.notna().any():
            return dt
        if 'data_original_str' in df.columns:
            dt2 = pd.to_datetime(df['data_original_str'], errors='coerce')
            if dt2.notna().any():
                return dt2
        if 'data_br' in df.columns:
            return pd.to_datetime(df['data_br'], dayfirst=True, errors='coerce')
        return dt

    @staticmethod
    def __num_series(df: pd.DataFrame, col: str) -> pd.Series:
        return pd.Series(pd.to_numeric(df[col], errors='coerce'), index=df.index)

    @staticmethod
    def __validate_schema(df: pd.DataFrame) -> None:
        required: List[str] = [
            'veiculo_id',
            'data',
            'km_dia_por_soma',
            'km_dia_por_odometro',
            'odo_ini_dia',
            'odo_fim_dia',
            'atividade_total_h',
            'viagens_qtd',
            'flag_reset_odometro',
        ]

        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f'Colunas obrigatorias ausentes: {missing}')

    @staticmethod
    def __sane_odometer_and_select(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values('data').copy()
        odo_fim: pd.Series = RelatorioService.__num_series(df, 'odo_fim_dia')
        odo_ini: pd.Series = RelatorioService.__num_series(df, 'odo_ini_dia')
        delta_raw: pd.Series = odo_fim - odo_ini
        delta_odo: pd.Series = delta_raw.copy()

        if 'flag_reset_odometro' in df.columns:
            reset_mask: pd.Series = RelatorioService.__num_series(df, 'flag_reset_odometro').fillna(0).astype('Int64') == 1
            delta_odo = delta_odo.mask(reset_mask)
        delta_odo = delta_odo.mask(delta_odo < 0)

        if 'km_dia_por_odometro' in df.columns:
            odo_pref: pd.Series = RelatorioService.__num_series(df, 'km_dia_por_odometro')
            odo_pref = odo_pref.where(odo_pref.notna(), delta_odo)
        else:
            odo_pref = delta_odo

        soma: pd.Series = RelatorioService.__num_series(df, 'km_dia_por_soma')
        km_sel: pd.Series = odo_pref.where(odo_pref.notna(), soma).fillna(0.0).clip(0, None)
        df['km_dia_final'] = km_sel.astype('float64')

        km_fonte = pd.Series('zero', index=df.index, dtype='string')
        km_fonte = km_fonte.where(~soma.notna(), 'soma')
        km_fonte = km_fonte.where(~odo_pref.notna(), 'odometro')
        df['km_fonte'] = km_fonte

        h: pd.Series = RelatorioService.__num_series(df, 'atividade_total_h').fillna(0.0).clip(0, None)
        df['h_dia_final'] = h.astype('float64')

        df['delta_odo'] = delta_odo
        df['delta_odo_negativo_flag'] = delta_raw < 0
        df['delta_odo_nan_flag'] = delta_odo.isna()
        return df

    @staticmethod
    def __cap_iqr(series: pd.Series) -> pd.Series:
        s: pd.Series = pd.Series(pd.to_numeric(series, errors='coerce'), index=series.index)
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        if pd.isna(q1) or pd.isna(q3):
            return s.clip(0, None)
        iqr = q3 - q1
        if iqr == 0:
            return s.clip(0, None)
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        return s.clip(max(0.0, lo), hi)

    @staticmethod
    def __apply_outliers(df: pd.DataFrame) -> pd.DataFrame:
        km_before: pd.Series = RelatorioService.__num_series(df, 'km_dia_final').fillna(0.0)
        h_before: pd.Series = RelatorioService.__num_series(df, 'h_dia_final').fillna(0.0)

        km_cap: pd.Series = km_before.clip(None, 2400.0)
        h_cap: pd.Series = h_before.clip(None, 24.0)

        km_iqr: pd.Series = RelatorioService.__cap_iqr(km_cap)
        h_iqr: pd.Series = RelatorioService.__cap_iqr(h_cap)

        if len(df) >= 90:
            mu_km = float(km_iqr.mean())
            sd_km = float(km_iqr.std(ddof=0)) or 1.0
            z_km = (km_iqr - mu_km) / sd_km
            km_final = km_iqr.mask(z_km.abs() > 4, mu_km + 4 * np.sign(z_km) * sd_km).clip(0, None)

            mu_h = float(h_iqr.mean())
            sd_h = float(h_iqr.std(ddof=0)) or 1.0
            z_h = (h_iqr - mu_h) / sd_h
            h_final = h_iqr.mask(z_h.abs() > 4, mu_h + 4 * np.sign(z_h) * sd_h).clip(0, None)
        else:
            km_final = km_iqr.clip(0, None)
            h_final = h_iqr.clip(0, None)

        df['km_dia_clean'] = km_final.astype('float32')
        df['h_dia_clean'] = h_final.astype('float32')
        return df

    @staticmethod
    def __apply_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df['data'] = RelatorioService.__parse_date_with_fallback(df)
        df = df[df['data'].notna()].sort_values('data').copy()
        if df.empty:
            return df

        km: pd.Series = RelatorioService.__num_series(df, 'km_dia_clean').fillna(0).astype('float32')
        hh: pd.Series = RelatorioService.__num_series(df, 'h_dia_clean').fillna(0).astype('float32')

        df = RelatorioService.__add_rolling_features(df, km, hh)
        df = RelatorioService.__add_calendar_features(df)
        df = RelatorioService.__add_parado_features(df, km)

        if 'data_br' not in df.columns:
            df['data_br'] = pd.Series(pd.DatetimeIndex(df['data']).strftime('%d/%m/%Y').tolist(), index=df.index)

        df['km_dia_clean'] = km
        df['h_dia_clean'] = hh
        df['veiculo_id'] = df['veiculo_id'].astype('string')
        return df

    @staticmethod
    def __add_rolling_features(df: pd.DataFrame, km: pd.Series, hh: pd.Series) -> pd.DataFrame:
        for w in RelatorioService.__FEATURE_WINDOWS:
            df[f'mm_km_{w}'] = km.rolling(window=w, min_periods=1).mean().astype('float32')
            df[f'sm_km_{w}'] = km.rolling(window=w, min_periods=1).sum().astype('float32')
            df[f'mm_h_{w}'] = hh.rolling(window=w, min_periods=1).mean().astype('float32')
        return df

    @staticmethod
    def __add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
        df.drop(columns=['fimdesemana'], errors='ignore', inplace=True)

        dt_idx = pd.DatetimeIndex(df['data'])
        wd: pd.Series = pd.Series(dt_idx.weekday.tolist(), index=df.index).astype('int16')
        df['weekday'] = wd
        df['fimdesemana'] = (wd >= 5).astype('int8')

        mes: pd.Series = pd.Series(dt_idx.month.tolist(), index=df.index).astype('int8')
        df['mes'] = mes

        ang = (2 * np.pi * (wd.astype('float32') / np.float32(7.0))).astype('float32')
        df['sin_sem'] = ang.apply(np.sin).astype('float32')
        df['cos_sem'] = ang.apply(np.cos).astype('float32')
        return df

    @staticmethod
    def __add_parado_features(df: pd.DataFrame, km: pd.Series) -> pd.DataFrame:
        flag_parado: pd.Series = (km == 0).astype('int8')
        df['flag_parado'] = flag_parado

        start_seq: pd.Series = ((flag_parado == 1) & (flag_parado.shift(1).fillna(0).astype('int8') == 0)).astype('int8')
        bloco_id: pd.Series = start_seq.cumsum()
        df['_bloco_id'] = bloco_id
        seq: pd.Series = df.groupby('_bloco_id', sort=False)['flag_parado'].cumcount().add(1)
        df['dias_parado_seq'] = seq.where(flag_parado == 1, 0).astype('int16')
        df.drop(columns=['_bloco_id'], inplace=True)
        df['pct_parado_7d'] = flag_parado.rolling(window=7, min_periods=1).mean().astype('float32')
        df['pct_parado_14d'] = flag_parado.rolling(window=14, min_periods=1).mean().astype('float32')
        return df

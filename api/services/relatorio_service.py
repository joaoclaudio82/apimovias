import numpy as np
import pandas as pd
from datetime import date
from pathlib import Path
from typing import Optional, Set, Tuple, List, Callable
from starlette.responses import FileResponse
from api.repositories.relatorio_repository import RelatorioRepository
from api.settings import settings
from api.utils.path_utils import PathUtils

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
        existing_keys: Set[Tuple[int, str]] = cls._existing_keys()
        for veiculos in RelatorioRepository.get_veiculos_stream(id_start, id_end, data_ini, data_fim):
            df: pd.DataFrame = pd.DataFrame.from_records(veiculos)
            df = cls.__normalize_dataframe(df)
            df['__key'] = list(zip(df['veiculo_id'], df['data']))
            df = df[~df['__key'].isin(existing_keys)].drop(columns='__key')
            if df.empty:
                continue

            existing_keys.update(zip(df['veiculo_id'], df['data']))
            df.to_csv(path, mode='a', index=False, header=PathUtils.is_empty(path))

    @staticmethod
    def _existing_keys(path: Path = settings.CSV_PATH) -> Set[Tuple[int, str]]:
        if PathUtils.is_empty(path):
            return set()

        df = pd.read_csv(path, usecols=['veiculo_id', 'data'])
        df['data'] = df['data'].astype(str)
        return set(zip(df['veiculo_id'], df['data']))

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
        df['fimdesemana'] = df['data'].dt.weekday.isin([5, 6]).astype('Int64')
        df = df.sort_values(by=['veiculo_id', 'data']).reset_index(drop=True)
        df = RelatorioService.__apply_per_vehicle(df, RelatorioService.__sane_odometer_and_select)
        df = RelatorioService.__apply_per_vehicle(df, RelatorioService.__apply_outliers)
        df = RelatorioService.__apply_per_vehicle(df, RelatorioService.__apply_features)
        df['data'] = df['data'].dt.date.astype(str)
        df = df[RelatorioService.__OUTPUT_COLUMNS]
        return df

    @staticmethod
    def __apply_per_vehicle(df: pd.DataFrame, fn: Callable[[pd.DataFrame], pd.DataFrame]) -> pd.DataFrame:
        if 'veiculo_id' not in df.columns:
            return fn(df)
        return df.groupby('veiculo_id', group_keys=False).apply(fn)

    @staticmethod
    def __parse_date_with_fallback(df: pd.DataFrame) -> pd.Series:
        dt = pd.to_datetime(df.get('data'), errors='coerce')
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

        if [col for col in required if col not in df.columns]:
            raise ValueError(f'Colunas obrigatorias ausentes: {missing}')

    @staticmethod
    def __sane_odometer_and_select(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values('data').copy()
        odo_fim = pd.to_numeric(df.get('odo_fim_dia'), errors='coerce')
        odo_ini = pd.to_numeric(df.get('odo_ini_dia'), errors='coerce')
        delta_raw = odo_fim - odo_ini
        delta_odo = delta_raw.copy()

        if 'flag_reset_odometro' in df.columns:
            reset_mask = pd.to_numeric(df['flag_reset_odometro'], errors='coerce').fillna(0).astype('Int64') == 1
            delta_odo = delta_odo.mask(reset_mask)
        delta_odo = delta_odo.mask(delta_odo < 0)

        if 'km_dia_por_odometro' in df.columns:
            odo_pref = pd.to_numeric(df['km_dia_por_odometro'], errors='coerce')
            odo_pref = odo_pref.where(odo_pref.notna(), delta_odo)
        else:
            odo_pref = delta_odo

        soma = pd.to_numeric(df.get('km_dia_por_soma'), errors='coerce')
        km_sel = odo_pref.where(odo_pref.notna(), soma).fillna(0.0).clip(lower=0)
        df['km_dia_final'] = km_sel.astype('float64')

        df['km_fonte'] = pd.Series(
            np.where(odo_pref.notna(), 'odometro', np.where(soma.notna(), 'soma', 'zero')),
            index=df.index,
            dtype='string'
        )

        if 'atividade_total_h' in df.columns:
            h = pd.to_numeric(df['atividade_total_h'], errors='coerce').fillna(0.0).clip(lower=0)
            df['h_dia_final'] = h.astype('float64')
        else:
            df['h_dia_final'] = 0.0

        df['delta_odo'] = delta_odo
        df['delta_odo_negativo_flag'] = delta_raw < 0
        df['delta_odo_nan_flag'] = delta_odo.isna()
        return df

    @staticmethod
    def __cap_iqr(series: pd.Series) -> pd.Series:
        s = pd.to_numeric(series, errors='coerce')
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        if pd.isna(q1) or pd.isna(q3):
            return s.clip(lower=0)
        iqr = q3 - q1
        if iqr == 0:
            return s.clip(lower=0)
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        return s.clip(lower=max(0.0, lo), upper=hi)

    @staticmethod
    def __apply_outliers(df: pd.DataFrame) -> pd.DataFrame:
        km_before = pd.to_numeric(df.get('km_dia_final'), errors='coerce').fillna(0.0)
        h_before = pd.to_numeric(df.get('h_dia_final'), errors='coerce').fillna(0.0)

        km_cap = km_before.clip(upper=2400.0)
        h_cap = h_before.clip(upper=24.0)

        km_iqr = RelatorioService.__cap_iqr(km_cap)
        h_iqr = RelatorioService.__cap_iqr(h_cap)

        if len(df) >= 90:
            mu_km = float(km_iqr.mean())
            sd_km = float(km_iqr.std(ddof=0)) or 1.0
            z_km = (km_iqr - mu_km) / sd_km
            km_final = km_iqr.mask(z_km.abs() > 4, mu_km + 4 * np.sign(z_km) * sd_km).clip(lower=0)

            mu_h = float(h_iqr.mean())
            sd_h = float(h_iqr.std(ddof=0)) or 1.0
            z_h = (h_iqr - mu_h) / sd_h
            h_final = h_iqr.mask(z_h.abs() > 4, mu_h + 4 * np.sign(z_h) * sd_h).clip(lower=0)
        else:
            km_final = km_iqr.clip(lower=0)
            h_final = h_iqr.clip(lower=0)

        df['km_dia_clean'] = km_final.astype('float32')
        df['h_dia_clean'] = h_final.astype('float32')
        return df

    @staticmethod
    def __apply_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df['data'] = RelatorioService.__parse_date_with_fallback(df)
        df = df[df['data'].notna()].sort_values('data')
        if df.empty:
            return df

        km = pd.to_numeric(df.get('km_dia_clean'), errors='coerce').fillna(0).astype('float32')
        hh = pd.to_numeric(df.get('h_dia_clean'), errors='coerce').fillna(0).astype('float32')

        for w in RelatorioService.__FEATURE_WINDOWS:
            df[f'mm_km_{w}'] = km.rolling(window=w, min_periods=1).mean().astype('float32')
            df[f'sm_km_{w}'] = km.rolling(window=w, min_periods=1).sum().astype('float32')
            df[f'mm_h_{w}'] = hh.rolling(window=w, min_periods=1).mean().astype('float32')

        df.drop(columns=['fimdesemana'], errors='ignore', inplace=True)

        wd = df['data'].dt.weekday
        wd = wd.where(df['data'].notna(), -1).fillna(-1).astype('int16')
        df['weekday'] = wd
        df['fimdesemana'] = (wd >= 5).astype('int8')

        mes = df['data'].dt.month
        mes = mes.where(df['data'].notna(), 0).fillna(0).astype('int8')
        df['mes'] = mes

        wd_pos = np.where(wd.values >= 0, wd.values, 0).astype('float32')
        ang = (2 * np.pi * (wd_pos / np.float32(7.0))).astype('float32')
        df['sin_sem'] = np.sin(ang).astype('float32')
        df['cos_sem'] = np.cos(ang).astype('float32')

        flag_parado = (km == 0).astype('int8')
        df['flag_parado'] = flag_parado

        start_seq = ((flag_parado == 1) & (flag_parado.shift(1).fillna(0).astype('int8') == 0)).astype('int8')
        bloco_id = start_seq.cumsum()
        seq = df.groupby(bloco_id, sort=False)['flag_parado'].cumcount() + 1
        df['dias_parado_seq'] = np.where(flag_parado == 1, seq, 0).astype('int16')

        df['pct_parado_7d'] = flag_parado.rolling(window=7, min_periods=1).mean().astype('float32')
        df['pct_parado_14d'] = flag_parado.rolling(window=14, min_periods=1).mean().astype('float32')

        if 'data_br' not in df.columns:
            df['data_br'] = df['data'].dt.strftime('%d/%m/%Y')

        df['km_dia_clean'] = km
        df['h_dia_clean'] = hh
        df['veiculo_id'] = df['veiculo_id'].astype('string')
        return df

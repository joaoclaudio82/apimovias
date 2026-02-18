from datetime import date
from typing import Iterable, Optional, Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy import text
from api.database import engine

SQL = r'''
WITH params_input AS (
  SELECT
    p.vei_id,
    p.data_ini,
    p.data_fim
  FROM unnest(
    CAST(:vei_ids AS bigint[]),
    CAST(:data_ini_list AS date[]),
    CAST(:data_fim_list AS date[])
  ) AS p(vei_id, data_ini, data_fim)
),
params AS (
  SELECT
    p.vei_id,
    p.data_ini,
    p.data_fim,
    COALESCE(p.data_ini::timestamp, '-infinity'::timestamp) AS end_dh_ini,
    COALESCE((p.data_fim + 1)::timestamp, 'infinity'::timestamp) AS end_dh_fim_exclusive
  FROM params_input p
  JOIN public.veiculo v
    ON v.id = p.vei_id
),
global_bounds AS (
  SELECT
    MIN(p.end_dh_ini) AS min_end_dh,
    MAX(p.end_dh_fim_exclusive) AS max_end_dh
  FROM params p
),
trips_prefiltered AS (
  SELECT
    a.vei_id,
    (a.end_dh::date) AS data,
    a.distance_travel,
    a.start_odometer,
    a.end_odometer,
    a.timings_trip_time,
    a.timings_on_time,
    a.timings_work_time
  FROM trips.alltrips a
  JOIN params p
    ON p.vei_id = a.vei_id
   AND a.end_dh >= p.end_dh_ini
   AND a.end_dh < p.end_dh_fim_exclusive
  JOIN global_bounds g
    ON true
  WHERE a.end_dh IS NOT NULL
    AND a.start_dh IS NOT NULL
    AND a.end_dh >= a.start_dh
    AND a.end_dh >= g.min_end_dh
    AND a.end_dh < g.max_end_dh
),
day_agg AS (
  SELECT
    t.vei_id,
    t.data,
    MIN(t.start_odometer) AS odo_ini_dia,
    MAX(t.end_odometer) AS odo_fim_dia,
    COUNT(*) AS viagens_qtd,
    SUM(GREATEST(0, LEAST(COALESCE(t.distance_travel, t.end_odometer - t.start_odometer, 0), 1200))) AS dist_total_km,
    SUM(GREATEST(0, LEAST(COALESCE(t.timings_trip_time, t.timings_on_time, t.timings_work_time, 0), 86400))) AS atividade_total_seg
  FROM trips_prefiltered t
  GROUP BY t.vei_id, t.data
),
day_bounds AS (
  SELECT
    d.vei_id,
    MIN(data) AS min_data,
    MAX(data) AS max_data
  FROM day_agg d
  GROUP BY d.vei_id
),
range_dates AS (
  SELECT
    p.vei_id,
    COALESCE(p.data_ini, b.min_data) AS d_ini,
    COALESCE(p.data_fim, b.max_data) AS d_fim
  FROM params p
  LEFT JOIN day_bounds b
    ON b.vei_id = p.vei_id
),
calendar AS (
  SELECT r.vei_id, gs::date AS data
  FROM range_dates r
  CROSS JOIN LATERAL generate_series(r.d_ini, r.d_fim, interval '1 day') gs
  WHERE r.d_ini IS NOT NULL
    AND r.d_fim IS NOT NULL
    AND r.d_ini <= r.d_fim
),
joined AS (
  SELECT
    c.vei_id,
    c.data,
    d.odo_ini_dia,
    d.odo_fim_dia,
    COALESCE(d.viagens_qtd, 0)           AS viagens_qtd,
    COALESCE(d.dist_total_km, 0)         AS dist_total_km,
    COALESCE(d.atividade_total_seg, 0)   AS atividade_total_seg
  FROM calendar c
  LEFT JOIN day_agg d
    ON d.vei_id = c.vei_id AND d.data = c.data
),
ffill AS (
  SELECT
    j.*,
    SUM(CASE WHEN j.odo_fim_dia IS NOT NULL THEN 1 ELSE 0 END)
      OVER (PARTITION BY j.vei_id ORDER BY j.data
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS grp
  FROM joined j
),
ffilled AS (
  SELECT
    f.vei_id,
    f.data,
    MAX(f.odo_fim_dia) OVER (PARTITION BY f.vei_id, f.grp) AS odo_fim_ffill,
    f.odo_ini_dia,
    f.odo_fim_dia,
    f.viagens_qtd,
    f.dist_total_km,
    f.atividade_total_seg
  FROM ffill f
),
calc AS (
  SELECT
    x.*,
    LAG(x.odo_fim_ffill) OVER (PARTITION BY x.vei_id ORDER BY x.data) AS odo_prev
  FROM ffilled x
),
calc_with_km AS (
  SELECT
    c.*,
    (c.odo_fim_ffill - c.odo_prev) AS km_dia_por_odometro_raw
  FROM calc c
)
SELECT
  c.vei_id                          AS veiculo_id,
  c.data,
  c.viagens_qtd,
  c.dist_total_km         AS km_dia_por_soma,
  c.atividade_total_seg,
  (c.atividade_total_seg / 3600.0) AS atividade_total_h,
  c.odo_ini_dia,
  c.odo_fim_dia,
  c.odo_fim_ffill,
  c.km_dia_por_odometro_raw,
  CASE
    WHEN c.km_dia_por_odometro_raw IS NULL THEN false
    WHEN c.km_dia_por_odometro_raw < -50 THEN true
    WHEN c.km_dia_por_odometro_raw > 2800 THEN true
    ELSE false
  END AS flag_reset_odometro,
  CASE
    WHEN c.km_dia_por_odometro_raw IS NULL THEN 0
    WHEN c.km_dia_por_odometro_raw < 0 THEN 0
    ELSE LEAST(c.km_dia_por_odometro_raw, 2800)
  END AS km_dia_por_odometro
FROM calc_with_km c
ORDER BY c.vei_id, c.data;
'''

class RelatorioRepository:
    @staticmethod
    def __fetch_rows_for_ids(batch_vei_ids: List[int]) -> List[Dict[str, Any]]:
        if not batch_vei_ids:
            return []

        data_ini_list: List[Optional[date]] = [None] * len(batch_vei_ids)
        data_fim_list: List[Optional[date]] = [None] * len(batch_vei_ids)
        rows: List[Dict[str, Any]] = []
        for chunk in RelatorioRepository.get_veiculos_stream(
            batch_vei_ids,
            data_ini_list,
            data_fim_list,
            vehicles_per_chunk=len(batch_vei_ids)
        ):
            rows.extend(chunk)
        return rows

    @staticmethod
    def get_veiculos(veiculo_id: int, data_ini: Optional[date], data_fim: Optional[date]) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {
          "vei_ids": [int(veiculo_id)],
          "data_ini_list": [data_ini],
          "data_fim_list": [data_fim]
        }

        with engine.connect() as connection:
            result = connection.execute(text(SQL), params)
            return [dict(row) for row in result.mappings().all()]

    @staticmethod
    def get_veiculos_stream(
        vei_ids: List[int],
        data_ini_list: List[Optional[date]],
        data_fim_list: List[Optional[date]],
        vehicles_per_chunk: int = 1
    ) -> Iterable[List[Dict[str, Any]]]:
        if not vei_ids:
            return

        if not (len(vei_ids) == len(data_ini_list) == len(data_fim_list)):
            raise ValueError("Listas de parâmetros com tamanhos diferentes.")

        vehicles_per_chunk = max(1, int(vehicles_per_chunk))

        params: Dict[str, Any] = {
          "vei_ids": vei_ids,
          "data_ini_list": data_ini_list,
          "data_fim_list": data_fim_list
        }

        with engine.connect() as connection:
            result = connection.execution_options(stream_results=True).execute(text(SQL), params)
            rows_chunk: List[Dict[str, Any]] = []
            rows_by_vehicle: List[Dict[str, Any]] = []
            current_vehicle_id: Optional[int] = None
            vehicles_in_chunk = 0

            for row in result.mappings():
                row_dict = dict(row)
                row_vehicle_id = int(row_dict["veiculo_id"])

                if current_vehicle_id is None:
                    current_vehicle_id = row_vehicle_id
                elif row_vehicle_id != current_vehicle_id:
                    if rows_by_vehicle:
                        rows_chunk.extend(rows_by_vehicle)
                        vehicles_in_chunk += 1
                        rows_by_vehicle = []

                    if vehicles_in_chunk >= vehicles_per_chunk and rows_chunk:
                        yield rows_chunk
                        rows_chunk = []
                        vehicles_in_chunk = 0

                    current_vehicle_id = row_vehicle_id

                rows_by_vehicle.append(row_dict)

            if rows_by_vehicle:
                rows_chunk.extend(rows_by_vehicle)
                vehicles_in_chunk += 1

            if rows_chunk:
                yield rows_chunk

    @staticmethod
    def get_all_veiculos_stream(batch_size: int = 350, parallel_workers: int = 1) -> Iterable[List[Dict[str, Any]]]:
        batch_size = max(1, int(batch_size))
        parallel_workers = max(1, int(parallel_workers))

        with engine.connect() as connection:
            vei_ids = [int(row[0]) for row in connection.execute(text("SELECT id FROM public.veiculo ORDER BY id"))]

        if not vei_ids:
            return

        total = len(vei_ids)
        batches: List[List[int]] = []
        for start in range(0, total, batch_size):
            end = start + batch_size
            batches.append(vei_ids[start:end])

        worker_count = min(parallel_workers, len(batches))
        if worker_count <= 1:
            for batch_vei_ids in batches:
                rows = RelatorioRepository.__fetch_rows_for_ids(batch_vei_ids)
                if rows:
                    yield rows
            return

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(RelatorioRepository.__fetch_rows_for_ids, batch_vei_ids)
                for batch_vei_ids in batches
            ]
            try:
                for future in as_completed(futures):
                    rows = future.result()
                    if rows:
                        yield rows
            except Exception:
                for future in futures:
                    future.cancel()
                raise

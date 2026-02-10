from datetime import date
from typing import Iterable, Optional, Dict, List, Any
from sqlalchemy import text
from api.database import engine

SQL = r'''
WITH params AS (
  SELECT
    *
  FROM unnest(
    CAST(:vei_ids AS bigint[]),
    CAST(:data_ini_list AS date[]),
    CAST(:data_fim_list AS date[])
  ) AS p(vei_id, data_ini, data_fim)
),
trips_day AS (
  SELECT
    a.vei_id,
    (a.end_dh::date) AS data,
    GREATEST(0, LEAST(COALESCE(a.distance_travel, a.end_odometer - a.start_odometer, 0), 1200)) AS dist_km,
    GREATEST(0, LEAST(COALESCE(a.timings_trip_time, a.timings_on_time, a.timings_work_time, 0), 86400)) AS atividade_seg,
    a.start_odometer AS odo_ini_viagem,
    a.end_odometer   AS odo_fim_viagem
  FROM trips.alltrips a
  JOIN params p ON p.vei_id = a.vei_id
  WHERE a.end_dh IS NOT NULL
    AND a.start_dh IS NOT NULL
    AND a.end_dh >= a.start_dh
    AND (p.data_ini IS NULL OR a.end_dh::date >= p.data_ini)
    AND (p.data_fim IS NULL OR a.end_dh::date <= p.data_fim)
),
day_agg AS (
  SELECT
    vei_id,
    data,
    MIN(odo_ini_viagem) AS odo_ini_dia,
    MAX(odo_fim_viagem) AS odo_fim_dia,
    COUNT(*)            AS viagens_qtd,
    SUM(dist_km)        AS dist_total_km,
    MAX(dist_km)        AS dist_max_km,
    MIN(dist_km)        AS dist_min_km,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY dist_km) AS dist_mediana_km,
    STDDEV_SAMP(dist_km) AS dist_desvio_padrao_km,
    SUM(atividade_seg)   AS atividade_total_seg
  FROM trips_day
  GROUP BY vei_id, data
),
range_dates AS (
  SELECT
    p.vei_id,
    COALESCE(p.data_ini, (SELECT MIN(data) FROM day_agg WHERE vei_id = p.vei_id), CURRENT_DATE - 30) AS d_ini,
    COALESCE(p.data_fim, (SELECT MAX(data) FROM day_agg WHERE vei_id = p.vei_id), CURRENT_DATE)       AS d_fim
  FROM params p
),
calendar AS (
  SELECT r.vei_id, gs::date AS data
  FROM range_dates r
  CROSS JOIN LATERAL generate_series(r.d_ini, r.d_fim, interval '1 day') gs
),
joined AS (
  SELECT
    c.vei_id,
    c.data,
    d.odo_ini_dia,
    d.odo_fim_dia,
    COALESCE(d.viagens_qtd, 0)           AS viagens_qtd,
    COALESCE(d.dist_total_km, 0)         AS dist_total_km,
    COALESCE(d.dist_max_km, 0)           AS dist_max_km,
    COALESCE(d.dist_min_km, 0)           AS dist_min_km,
    COALESCE(d.dist_mediana_km, 0)       AS dist_mediana_km,
    COALESCE(d.dist_desvio_padrao_km, 0) AS dist_desvio_padrao_km,
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
    f.dist_max_km,
    f.dist_min_km,
    f.dist_mediana_km,
    f.dist_desvio_padrao_km,
    f.atividade_total_seg
  FROM ffill f
),
calc AS (
  SELECT
    x.*,
    LAG(x.odo_fim_ffill) OVER (PARTITION BY x.vei_id ORDER BY x.data) AS odo_prev,
    (x.odo_fim_ffill - LAG(x.odo_fim_ffill) OVER (PARTITION BY x.vei_id ORDER BY x.data)) AS km_dia_por_odometro_raw
  FROM ffilled x
)
SELECT
  v.id                              AS veiculo_id,
  v.ano_fabricacao,
  v.ano_modelo,
  v.modelo_veiculo_id,
  v.marca_veiculo_id,
  v.tipo_veiculo_id,
  v.tipo_combustivel,
  c.data,
  c.viagens_qtd,
  c.dist_total_km         AS km_dia_por_soma,
  c.dist_max_km,
  c.dist_min_km,
  c.dist_mediana_km,
  c.dist_desvio_padrao_km,
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
FROM calc c
JOIN public.veiculo v
  ON v.id = c.vei_id
ORDER BY c.vei_id, c.data;
'''

class RelatorioRepository:
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
    def get_veiculos_stream(vei_ids: List[int], data_ini_list: List[Optional[date]], data_fim_list: List[Optional[date]]) -> Iterable[List[Dict[str, Any]]]:
        if not vei_ids:
            return

        if not (len(vei_ids) == len(data_ini_list) == len(data_fim_list)):
            raise ValueError("Listas de parâmetros com tamanhos diferentes.")

        params: Dict[str, Any] = {
          "vei_ids": vei_ids,
          "data_ini_list": data_ini_list,
          "data_fim_list": data_fim_list
        }

        with engine.connect() as connection:
            result = connection.execute(text(SQL), params)
            rows = [dict(row) for row in result.mappings().all()]

        if not rows:
            return

        by_vehicle: Dict[int, List[Dict[str, Any]]] = {}
        for row in rows:
            by_vehicle.setdefault(int(row["veiculo_id"]), []).append(row)

        for veiculo_id in vei_ids:
            veiculos = by_vehicle.get(veiculo_id)
            if veiculos:
                yield veiculos

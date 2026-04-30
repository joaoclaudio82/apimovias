#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import datetime as dt
import pandas as pd
from sqlalchemy import create_engine, text

# ------------------------------------------------------------
# CONFIG (Azure PostgreSQL)
# ------------------------------------------------------------
PGHOST = "movias-pgsql.postgres.database.azure.com"
PGPORT = 5432
PGDATABASE = "moviasfleet"
PGUSER = "moviasuece"
PGPASSWORD = "894hGWcvPs4k"  # cuidado: credencial sensível

CSV_PATH = "resultado_movias.csv"
DATA_INI = dt.date(2025, 1, 1)  # <- date (não string)
DATA_FIM = dt.date(2025, 10, 10)  # <- date (não string)

ID_START = 1
ID_END = 29600  # inclusivo

# Consulta (mesma lógica, usando parâmetros nomeados)
SQL = r"""
WITH params AS (
  SELECT 
    CAST(:vei_id   AS bigint) AS vei_id,
    CAST(:data_ini AS date)   AS data_ini,
    CAST(:data_fim AS date)   AS data_fim
),

-- 1) Saneia viagens e prepara bases por dia
trips_day AS (
  SELECT
    a.vei_id,
    (a.end_dh::date) AS data,
    -- distância por viagem, saneada [0,1200] km
    GREATEST(0, LEAST(COALESCE(a.distance_travel, a.end_odometer - a.start_odometer, 0), 1200)) AS dist_km,
    -- tempo de atividade (segundos), saneado [0,86400]
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

-- 2) Agrega por dia (odômetro do dia + estatísticas)
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

-- 3) Faixa de datas (fallback: 30 dias)
range_dates AS (
  SELECT
    p.vei_id,
    COALESCE(p.data_ini, (SELECT MIN(data) FROM day_agg WHERE vei_id = p.vei_id), CURRENT_DATE - 30) AS d_ini,
    COALESCE(p.data_fim, (SELECT MAX(data) FROM day_agg WHERE vei_id = p.vei_id), CURRENT_DATE)       AS d_fim
  FROM params p
),

-- 4) Calendário contínuo
calendar AS (
  SELECT r.vei_id, gs::date AS data
  FROM range_dates r
  CROSS JOIN LATERAL generate_series(r.d_ini, r.d_fim, interval '1 day') gs
),

-- 5) Junta calendário com agregados (dias sem viagem vêm como NULL e viram 0)
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

-- 6) Forward-fill do odômetro final do dia
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
    -- odômetro final do dia (com ffill)
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

-- 7) km por odômetro via LAG + flags de reset/outlier
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

  -- distâncias por soma das viagens (já saneadas)
  c.dist_total_km         AS km_dia_por_soma,
  c.dist_max_km,
  c.dist_min_km,
  c.dist_mediana_km,
  c.dist_desvio_padrao_km,

  -- tempos
  c.atividade_total_seg,
  (c.atividade_total_seg / 3600.0) AS atividade_total_h,

  -- odômetro do dia
  c.odo_ini_dia,
  c.odo_fim_dia,
  c.odo_fim_ffill,

  -- km por LAG do odômetro (bruto e tratado)
  c.km_dia_por_odometro_raw,
  -- flag de reset/outlier: queda grande ou salto absurdo
  CASE 
    WHEN c.km_dia_por_odometro_raw IS NULL THEN false
    WHEN c.km_dia_por_odometro_raw < -50 THEN true
    WHEN c.km_dia_por_odometro_raw > 2800 THEN true
    ELSE false
  END AS flag_reset_odometro,
  -- versão tratada (clip para [0,2800]; zera se reset)
  CASE 
    WHEN c.km_dia_por_odometro_raw IS NULL THEN 0
    WHEN c.km_dia_por_odometro_raw < 0 THEN 0
    ELSE LEAST(c.km_dia_por_odometro_raw, 2800)
  END AS km_dia_por_odometro

FROM calc c
JOIN public.veiculo v
  ON v.id = c.vei_id
ORDER BY c.data;
"""


def main():
    # engine com SSL
    engine = create_engine(
        f"postgresql+psycopg2://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}",
        connect_args={"sslmode": "require"},
        pool_pre_ping=True,
    )

    # Verifica se o arquivo já existe (para decidir header no primeiro append)
    file_exists = os.path.isfile(CSV_PATH)

    total_adicionados = 0
    total_ids_sem_dados = 0

    try:
        for vid in range(ID_START, ID_END + 1):
            params = {
                "vei_id": int(vid),
                "data_ini": DATA_INI,
                "data_fim": DATA_FIM,
            }

            try:
                df = pd.read_sql_query(text(SQL), engine, params=params)
            except Exception as e:
                print(f"[ERRO] id={vid}: falha na consulta ({e})")
                continue

            if df.empty:
                print(f"[AVISO] id={vid}: nenhum registro no período {DATA_INI} a {DATA_FIM}.")
                total_ids_sem_dados += 1
                continue

            # Normalizações que você fazia antes (se desejar manter)
            df["data"] = pd.to_datetime(df["data"])
            df["fimdesemana"] = df["data"].dt.weekday.isin([5, 6])
            cols = [c for c in df.columns if c != "fimdesemana"] + ["fimdesemana"]
            df = df[cols].sort_values(by=["veiculo_id", "data"]).reset_index(drop=True)

            # Append no CSV (cabeçalho só se o arquivo ainda não existe)
            try:
                df.to_csv(
                    CSV_PATH,
                    mode="a",
                    index=False,
                    header=not file_exists,
                    encoding="utf-8",
                )
                file_exists = True  # após a primeira escrita, sempre false para header
                print(f"[OK] id={vid}: {len(df)} linha(s) adicionada(s) ao arquivo.")
                total_adicionados += 1
            except Exception as e:
                print(f"[ERRO] id={vid}: falha ao gravar no CSV ({e})")

    finally:
        # fecha conexões
        try:
            engine.dispose()
        except Exception:
            pass

    print("----------------------------------------------------")
    print(f"Processo concluído. Arquivo: {CSV_PATH}")
    print(f"IDs com dados adicionados: {total_adicionados}")
    print(f"IDs sem dados no período: {total_ids_sem_dados}")
    print("----------------------------------------------------")


if __name__ == "__main__":
    main()

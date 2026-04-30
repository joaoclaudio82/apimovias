CREATE SCHEMA IF NOT EXISTS trips;

CREATE TABLE IF NOT EXISTS trips.alltrips (
  id BIGSERIAL PRIMARY KEY,
  vei_id BIGINT NOT NULL,
  start_dh TIMESTAMP NOT NULL,
  end_dh TIMESTAMP NOT NULL,
  distance_travel NUMERIC,
  start_odometer NUMERIC,
  end_odometer NUMERIC,
  timings_trip_time INTEGER,
  timings_on_time INTEGER,
  timings_work_time INTEGER
);

CREATE TABLE IF NOT EXISTS public.veiculo (
  id BIGINT PRIMARY KEY,
  ano_fabricacao INTEGER,
  ano_modelo INTEGER,
  modelo_veiculo_id INTEGER,
  marca_veiculo_id INTEGER,
  tipo_veiculo_id INTEGER,
  tipo_combustivel TEXT
);

INSERT INTO public.veiculo (
  id, ano_fabricacao, ano_modelo, modelo_veiculo_id, marca_veiculo_id, tipo_veiculo_id, tipo_combustivel
) VALUES
  (1, 2022, 2023, 101, 10, 1, 'diesel'),
  (2, 2021, 2022, 102, 11, 2, 'gasolina')
ON CONFLICT (id) DO NOTHING;

INSERT INTO trips.alltrips (
  vei_id, start_dh, end_dh, distance_travel, start_odometer, end_odometer,
  timings_trip_time, timings_on_time, timings_work_time
) VALUES
  (1, '2025-01-01 08:10:00', '2025-01-01 09:05:00', 42.5, 10000, 10042.5, 3300, 3300, 3400),
  (1, '2025-01-01 12:00:00', '2025-01-01 12:30:00', 18.0, 10042.5, 10060.5, 1800, 1800, 1900),
  (1, '2025-01-02 07:00:00', '2025-01-02 08:20:00', 65.0, 10060.5, 10125.5, 4800, 4800, 5000),
  (1, '2025-01-04 09:15:00', '2025-01-04 10:10:00', 55.0, 10125.5, 10180.5, 3300, 3300, 3400),
  (2, '2025-01-02 06:30:00', '2025-01-02 07:10:00', 30.0, 5000, 5030, 2400, 2400, 2400),
  (2, '2025-01-03 18:00:00', '2025-01-03 19:00:00', 50.0, 5030, 5080, 3600, 3600, 3800),
  (2, '2025-01-05 08:00:00', '2025-01-05 09:30:00', 75.0, 5080, 5155, 5400, 5400, 5600);

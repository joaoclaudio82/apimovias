"""
Faz backup da base de dados e limpa as tabelas de ingestão/predição.

Tabelas limpas:
  - daily_activity
  - daily_activity_removed
  - vehicle_profile
  - vehicle_metadata_h
  - vehicle_metadata_km
  - profile_metadata
  - predictions_daily
  - predictions_heads

Tabelas preservadas:
  - active_models
  - pipeline_runs
  - alembic_version

Uso:
    python reset_ingestion.py            # backup + limpeza
    python reset_ingestion.py --dry-run  # mostra o que seria feito
"""

import argparse
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "database.db"

INGESTION_TABLES = [
    "daily_activity",
    "daily_activity_removed",
    "vehicle_profile",
    "vehicle_metadata_h",
    "vehicle_metadata_km",
    "profile_metadata",
    "predictions_daily",
    "predictions_heads",
]


def backup(db_path: Path) -> Path:
    """Copia a base de dados para um ficheiro com timestamp."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.with_name(f"database_backup_{ts}.db")
    shutil.copy2(str(db_path), str(backup_path))
    return backup_path


def reset(db_path: Path, dry_run: bool = False):
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    for table in INGESTION_TABLES:
        cur.execute(f"SELECT COUNT(*) FROM [{table}]")
        count = cur.fetchone()[0]
        action = "APAGARIA" if dry_run else "DELETE FROM"
        print(f"  {table}: {count} registos {'(dry-run)' if dry_run else ''}")
        if not dry_run and count > 0:
            cur.execute(f"DELETE FROM [{table}]")

    if not dry_run:
        conn.commit()
        conn.execute("VACUUM")
        print("\nTabelas limpas e base compactada.")
    else:
        print("\n(dry-run — nenhuma alteração feita)")

    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Reset tabelas de ingestão")
    parser.add_argument("--dry-run", action="store_true", help="Apenas mostrar o que seria feito")
    args = parser.parse_args()

    if not DB_PATH.exists():
        print(f"Base de dados não encontrada: {DB_PATH}")
        return

    size_mb = DB_PATH.stat().st_size / (1024 * 1024)
    print(f"Base de dados: {DB_PATH} ({size_mb:.1f} MB)")

    if not args.dry_run:
        bkp = backup(DB_PATH)
        print(f"Backup criado: {bkp}\n")
    else:
        print("(dry-run — sem backup)\n")

    reset(DB_PATH, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
import sqlite3
import pandas as pd

from .utils import load_config


RAW_TABLES = {
    "products": "products.csv",
    "regions": "regions.csv",
    "warehouses": "warehouses.csv",
    "shipping_cost_matrix": "shipping_cost_matrix.csv",
    "supplier_lead_times": "supplier_lead_times.csv",
    "orders": "orders.csv",
    "shipments": "shipments.csv",
    "inventory_snapshot": "inventory_snapshot.csv",
}


def build_sqlite_database(root: Path) -> None:
    cfg = load_config(root)
    db_path = root / cfg["database_name"]
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    raw_dir = root / "data" / "raw"

    for table, file_name in RAW_TABLES.items():
        df = pd.read_csv(raw_dir / file_name)
        df.to_sql(table, conn, index=False, if_exists="replace")

    # Create raw-layer views. Processed tables are added later by reporting.py.
    views_path = root / "sql" / "views.sql"
    if views_path.exists():
        conn.executescript(views_path.read_text(encoding="utf-8"))

    conn.close()
    print(f"  Created database: {db_path.name}")


def read_sql(root: Path, query: str) -> pd.DataFrame:
    cfg = load_config(root)
    conn = sqlite3.connect(root / cfg["database_name"])
    try:
        return pd.read_sql_query(query, conn)
    finally:
        conn.close()

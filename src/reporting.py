from __future__ import annotations

from pathlib import Path
import sqlite3
import pandas as pd

from .utils import load_config


def build_powerbi_outputs(root: Path) -> None:
    cfg = load_config(root)
    db_path = root / cfg["database_name"]
    processed_dir = root / "data" / "processed"
    out_dir = root / "data" / "powerbi_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    inv = pd.read_csv(processed_dir / "inventory_recommendations.csv")
    forecast_metrics = pd.read_csv(processed_dir / "forecast_metrics.csv")
    scenario = pd.read_csv(processed_dir / "scenario_results.csv")
    network = pd.read_csv(processed_dir / "network_recommendations.csv")
    backtest = pd.read_csv(processed_dir / "forecast_backtest.csv")
    feature_importance = pd.read_csv(processed_dir / "forecast_feature_importance.csv")
    scenario_detail = pd.read_csv(processed_dir / "scenario_detail.csv")
    orders = pd.read_csv(root / "data" / "raw" / "orders.csv", parse_dates=["order_date"])
    shipments = pd.read_csv(root / "data" / "raw" / "shipments.csv")

    executive = pd.DataFrame([{
        "total_orders": len(orders),
        "total_revenue": round(orders["revenue"].sum(), 2),
        "forecast_mape_pct": forecast_metrics.loc[0, "mape_pct"],
        "forecast_bias_pct": forecast_metrics.loc[0, "forecast_bias_pct"],
        "stockout_risk_pairs": int((inv["stockout_risk"] == "High").sum()),
        "overstock_risk_pairs": int((inv["overstock_risk"] == "High").sum()),
        "recommended_order_units": int(inv["recommended_order_units"].sum()),
        "estimated_replenishment_cost": round(inv["estimated_replenishment_cost"].sum(), 2),
        "estimated_future_shipping_cost": round(network["estimated_shipping_cost"].sum(), 2),
        "late_delivery_rate_pct": round(shipments["late_delivery_flag"].mean() * 100, 2),
    }])
    executive.to_csv(out_dir / "executive_summary.csv", index=False)

    monthly_kpis = (
        orders.merge(shipments[["order_id", "late_delivery_flag", "shipping_cost"]], on="order_id", how="left")
        .assign(month=lambda x: x["order_date"].dt.to_period("M").astype(str))
        .groupby("month", as_index=False)
        .agg(
            orders=("order_id", "nunique"),
            units=("order_units", "sum"),
            revenue=("revenue", "sum"),
            shipping_cost=("shipping_cost", "sum"),
            late_delivery_rate=("late_delivery_flag", "mean"),
        )
    )
    monthly_kpis["late_delivery_rate_pct"] = (monthly_kpis["late_delivery_rate"] * 100).round(2)
    monthly_kpis["revenue"] = monthly_kpis["revenue"].round(2)
    monthly_kpis["shipping_cost"] = monthly_kpis["shipping_cost"].round(2)
    monthly_kpis.to_csv(out_dir / "monthly_kpis.csv", index=False)

    category_risk = (
        inv.groupby("category", as_index=False)
        .agg(
            sku_warehouse_pairs=("sku_id", "count"),
            high_stockout_pairs=("stockout_risk", lambda s: int((s == "High").sum())),
            high_overstock_pairs=("overstock_risk", lambda s: int((s == "High").sum())),
            recommended_units=("recommended_order_units", "sum"),
            replenishment_cost=("estimated_replenishment_cost", "sum"),
            avg_stock_coverage_weeks=("stock_coverage_weeks", "mean"),
        )
    )
    category_risk["replenishment_cost"] = category_risk["replenishment_cost"].round(2)
    category_risk["avg_stock_coverage_weeks"] = category_risk["avg_stock_coverage_weeks"].round(2)
    category_risk.to_csv(out_dir / "category_inventory_risk.csv", index=False)

    # Export core tables for Power BI.
    inv.to_csv(out_dir / "inventory_recommendations.csv", index=False)
    scenario.to_csv(out_dir / "scenario_results.csv", index=False)
    scenario_detail.to_csv(out_dir / "scenario_detail.csv", index=False)
    network.to_csv(out_dir / "network_recommendations.csv", index=False)
    backtest.to_csv(out_dir / "forecast_accuracy_detail.csv", index=False)
    feature_importance.head(30).to_csv(out_dir / "forecast_feature_importance_top30.csv", index=False)

    # Also write processed tables into SQLite so Power BI can connect to one local DB.
    conn = sqlite3.connect(db_path)
    for name, df in {
        "forecast_output": pd.read_csv(processed_dir / "forecast_output.csv"),
        "forecast_metrics": forecast_metrics,
        "forecast_backtest": backtest,
        "forecast_feature_importance": feature_importance,
        "inventory_recommendations": inv,
        "scenario_results": scenario,
        "scenario_detail": scenario_detail,
        "network_recommendations": network,
        "executive_summary": executive,
        "monthly_kpis": monthly_kpis,
        "category_inventory_risk": category_risk,
    }.items():
        df.to_sql(name, conn, index=False, if_exists="replace")

    # Create processed-layer views after tables exist.
    processed_views = root / "sql" / "processed_views.sql"
    if processed_views.exists():
        conn.executescript(processed_views.read_text(encoding="utf-8"))

    conn.close()

    print(f"  Power BI-ready outputs written to {out_dir}.")

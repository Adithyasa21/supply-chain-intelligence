from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .utils import service_level_to_z


def run_inventory_optimization(root: Path) -> None:
    processed_dir = root / "data" / "processed"

    forecast = pd.read_csv(processed_dir / "forecast_output.csv", parse_dates=["week_start"])
    orders = pd.read_csv(root / "data" / "raw" / "orders.csv", parse_dates=["week_start"])
    products = pd.read_csv(root / "data" / "raw" / "products.csv")
    inventory = pd.read_csv(root / "data" / "raw" / "inventory_snapshot.csv")
    lead_times = pd.read_csv(root / "data" / "raw" / "supplier_lead_times.csv")

    # Map future regional demand to warehouses using historical fulfillment shares.
    hist_share = orders.groupby(["sku_id", "region_id", "warehouse_id"], as_index=False)["order_units"].sum()
    totals = hist_share.groupby(["sku_id", "region_id"])["order_units"].transform("sum")
    hist_share["fulfillment_share"] = hist_share["order_units"] / totals.replace(0, 1)

    future_wh = forecast.merge(
        hist_share[["sku_id", "region_id", "warehouse_id", "fulfillment_share"]],
        on=["sku_id", "region_id"],
        how="left",
    )
    future_wh["fulfillment_share"] = future_wh["fulfillment_share"].fillna(0)
    future_wh["warehouse_forecast_units"] = future_wh["forecasted_demand"] * future_wh["fulfillment_share"]

    forecast_by_sku_wh = (
        future_wh.groupby(["sku_id", "warehouse_id"], as_index=False)
        .agg(
            forecast_horizon_units=("warehouse_forecast_units", "sum"),
            avg_weekly_forecast_units=("warehouse_forecast_units", "mean"),
        )
    )

    hist_weekly = orders.groupby(["sku_id", "warehouse_id", "week_start"], as_index=False)["order_units"].sum()
    demand_stats = (
        hist_weekly.groupby(["sku_id", "warehouse_id"], as_index=False)
        .agg(
            avg_weekly_demand=("order_units", "mean"),
            std_weekly_demand=("order_units", "std"),
            annualized_demand=("order_units", lambda s: float(s.mean() * 52)),
        )
    )
    demand_stats["std_weekly_demand"] = demand_stats["std_weekly_demand"].fillna(0)

    recs = (
        forecast_by_sku_wh
        .merge(demand_stats, on=["sku_id", "warehouse_id"], how="left")
        .merge(inventory, on=["sku_id", "warehouse_id"], how="left")
        .merge(lead_times, on=["sku_id", "warehouse_id"], how="left")
        .merge(products, on="sku_id", how="left")
    )

    recs["avg_weekly_demand"] = recs["avg_weekly_demand"].fillna(recs["avg_weekly_forecast_units"]).fillna(0)
    recs["std_weekly_demand"] = recs["std_weekly_demand"].fillna(0)
    recs["annualized_demand"] = recs["annualized_demand"].fillna(recs["avg_weekly_forecast_units"] * 52).fillna(0)
    recs["current_inventory_units"] = recs["current_inventory_units"].fillna(0)
    recs["reserved_inventory_units"] = recs["reserved_inventory_units"].fillna(0)
    recs["available_inventory_units"] = (recs["current_inventory_units"] - recs["reserved_inventory_units"]).clip(lower=0)
    recs["avg_supplier_lead_time_days"] = recs["avg_supplier_lead_time_days"].fillna(10)
    recs["lead_time_weeks"] = recs["avg_supplier_lead_time_days"] / 7

    recs["z_score"] = recs["target_service_level"].apply(service_level_to_z)
    recs["lead_time_demand_units"] = recs["avg_weekly_forecast_units"] * recs["lead_time_weeks"]
    recs["safety_stock_units"] = recs["z_score"] * recs["std_weekly_demand"] * np.sqrt(recs["lead_time_weeks"].clip(lower=0.01))
    recs["reorder_point_units"] = recs["lead_time_demand_units"] + recs["safety_stock_units"]

    annual_holding_cost_per_unit = (recs["unit_cost"] * recs["holding_cost_pct"]).replace(0, 1)
    recs["eoq_units"] = np.sqrt((2 * recs["annualized_demand"].clip(lower=1) * recs["order_cost"]) / annual_holding_cost_per_unit)

    recs["stock_coverage_weeks"] = recs["available_inventory_units"] / recs["avg_weekly_forecast_units"].replace(0, np.nan)
    recs["stock_coverage_weeks"] = recs["stock_coverage_weeks"].replace([np.inf, -np.inf], np.nan).fillna(999)
    recs["inventory_turnover"] = recs["annualized_demand"] / recs["current_inventory_units"].replace(0, np.nan)
    recs["inventory_turnover"] = recs["inventory_turnover"].replace([np.inf, -np.inf], np.nan).fillna(0)

    recs["shortage_units"] = (recs["reorder_point_units"] - recs["available_inventory_units"]).clip(lower=0)
    recs["excess_units"] = (recs["available_inventory_units"] - (recs["reorder_point_units"] + recs["eoq_units"])).clip(lower=0)

    recs["stockout_risk"] = np.where(recs["available_inventory_units"] < recs["reorder_point_units"], "High", "Low")
    recs["overstock_risk"] = np.where(recs["excess_units"] > recs["avg_weekly_forecast_units"] * 4, "High", "Low")
    recs["recommended_order_units"] = np.where(
        recs["stockout_risk"] == "High",
        np.ceil(recs["eoq_units"] + recs["shortage_units"]),
        0,
    ).astype(int)

    recs["estimated_replenishment_cost"] = (recs["recommended_order_units"] * recs["unit_cost"]).round(2)
    recs["estimated_holding_cost_annual"] = (recs["current_inventory_units"] * recs["unit_cost"] * recs["holding_cost_pct"]).round(2)

    keep_cols = [
        "sku_id", "product_name", "warehouse_id", "category", "velocity_class", "avg_weekly_forecast_units",
        "forecast_horizon_units", "available_inventory_units", "avg_supplier_lead_time_days",
        "target_service_level", "safety_stock_units", "reorder_point_units", "eoq_units",
        "stock_coverage_weeks", "inventory_turnover", "shortage_units", "excess_units",
        "stockout_risk", "overstock_risk", "recommended_order_units",
        "estimated_replenishment_cost", "estimated_holding_cost_annual",
    ]
    out = recs[keep_cols].copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].round(2)
    out.to_csv(processed_dir / "inventory_recommendations.csv", index=False)

    n_high = int((out["stockout_risk"] == "High").sum())
    print(f"  Inventory optimization completed. High stockout-risk SKU-warehouse pairs: {n_high:,}.")

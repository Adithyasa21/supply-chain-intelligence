from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def run_network_optimization(root: Path) -> None:
    """Greedy warehouse-to-region allocation under capacity.

    This is dependency-light so the project runs standalone. It mimics a network
    optimization model by assigning forecasted demand to the lowest-cost feasible
    warehouse while respecting weekly warehouse capacity.
    """
    processed_dir = root / "data" / "processed"
    forecast = pd.read_csv(processed_dir / "forecast_output.csv", parse_dates=["week_start"])
    costs = pd.read_csv(root / "data" / "raw" / "shipping_cost_matrix.csv")
    warehouses = pd.read_csv(root / "data" / "raw" / "warehouses.csv")

    weekly_demand = forecast.groupby(["week_start", "sku_id", "region_id"], as_index=False)["forecasted_demand"].sum()
    capacity = dict(zip(warehouses["warehouse_id"], warehouses["weekly_capacity_units"]))

    rows = []
    for week, week_df in weekly_demand.groupby("week_start"):
        remaining_capacity = {k: float(v) for k, v in capacity.items()}
        for _, demand_row in week_df.iterrows():
            region_costs = costs[costs["region_id"] == demand_row["region_id"]].sort_values("shipping_cost_per_unit")
            demand_remaining = float(demand_row["forecasted_demand"])

            for _, route in region_costs.iterrows():
                if demand_remaining <= 0:
                    break
                wh = route["warehouse_id"]
                available = remaining_capacity.get(wh, 0.0)
                assigned = min(demand_remaining, available)
                if assigned <= 0:
                    continue
                remaining_capacity[wh] -= assigned
                demand_remaining -= assigned
                rows.append({
                    "week_start": week.date().isoformat(),
                    "sku_id": demand_row["sku_id"],
                    "region_id": demand_row["region_id"],
                    "warehouse_id": wh,
                    "assigned_units": round(assigned, 2),
                    "shipping_cost_per_unit": route["shipping_cost_per_unit"],
                    "estimated_shipping_cost": round(assigned * route["shipping_cost_per_unit"], 2),
                    "base_transit_days": route["base_transit_days"],
                    "unmet_units": 0.0,
                })

            if demand_remaining > 0:
                rows.append({
                    "week_start": week.date().isoformat(),
                    "sku_id": demand_row["sku_id"],
                    "region_id": demand_row["region_id"],
                    "warehouse_id": "UNALLOCATED",
                    "assigned_units": 0.0,
                    "shipping_cost_per_unit": np.nan,
                    "estimated_shipping_cost": 0.0,
                    "base_transit_days": np.nan,
                    "unmet_units": round(demand_remaining, 2),
                })

    allocation = pd.DataFrame(rows)
    allocation.to_csv(processed_dir / "network_recommendations.csv", index=False)

    total_cost = allocation["estimated_shipping_cost"].sum()
    unmet = allocation["unmet_units"].sum()
    print(f"  Network allocation completed. Estimated shipping cost=${total_cost:,.0f}, unmet units={unmet:,.0f}.")

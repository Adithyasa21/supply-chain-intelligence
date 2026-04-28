from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .utils import service_level_to_z


SCENARIOS = [
    {"scenario_name": "Base Case", "demand_multiplier": 1.00, "lead_time_days_added": 0, "service_level": None, "shipping_cost_multiplier": 1.00, "capacity_multiplier": 1.00},
    {"scenario_name": "Demand Surge +20%", "demand_multiplier": 1.20, "lead_time_days_added": 0, "service_level": None, "shipping_cost_multiplier": 1.00, "capacity_multiplier": 1.00},
    {"scenario_name": "Supplier Delay +5 Days", "demand_multiplier": 1.00, "lead_time_days_added": 5, "service_level": None, "shipping_cost_multiplier": 1.00, "capacity_multiplier": 1.00},
    {"scenario_name": "Higher Service Level 97%", "demand_multiplier": 1.00, "lead_time_days_added": 0, "service_level": 0.97, "shipping_cost_multiplier": 1.00, "capacity_multiplier": 1.00},
    {"scenario_name": "Shipping Cost +10%", "demand_multiplier": 1.00, "lead_time_days_added": 0, "service_level": None, "shipping_cost_multiplier": 1.10, "capacity_multiplier": 1.00},
    {"scenario_name": "Warehouse Capacity -15%", "demand_multiplier": 1.00, "lead_time_days_added": 0, "service_level": None, "shipping_cost_multiplier": 1.00, "capacity_multiplier": 0.85},
]


def run_scenario_simulation(root: Path) -> None:
    processed_dir = root / "data" / "processed"
    inv = pd.read_csv(processed_dir / "inventory_recommendations.csv")
    network = pd.read_csv(processed_dir / "network_recommendations.csv")
    warehouses = pd.read_csv(root / "data" / "raw" / "warehouses.csv")

    results = []
    details = []

    for scenario in SCENARIOS:
        scenario_inv = inv.copy()
        scenario_inv["scenario_name"] = scenario["scenario_name"]
        scenario_inv["scenario_avg_weekly_forecast"] = scenario_inv["avg_weekly_forecast_units"] * scenario["demand_multiplier"]

        if scenario["service_level"] is None:
            scenario_inv["scenario_z_score"] = scenario_inv["target_service_level"].apply(service_level_to_z)
            service_label = "product_default"
        else:
            scenario_inv["scenario_z_score"] = service_level_to_z(scenario["service_level"])
            service_label = scenario["service_level"]

        lead_time_weeks = (scenario_inv["avg_supplier_lead_time_days"] + scenario["lead_time_days_added"]) / 7
        original_z = scenario_inv["target_service_level"].apply(service_level_to_z).replace(0, 1)
        original_lt = (scenario_inv["avg_supplier_lead_time_days"] / 7).clip(lower=0.01)
        estimated_std = scenario_inv["safety_stock_units"] / (original_z * np.sqrt(original_lt))
        estimated_std = estimated_std.replace([np.inf, -np.inf], 0).fillna(0)

        scenario_inv["scenario_safety_stock"] = scenario_inv["scenario_z_score"] * estimated_std * np.sqrt(lead_time_weeks.clip(lower=0.01))
        scenario_inv["scenario_reorder_point"] = scenario_inv["scenario_avg_weekly_forecast"] * lead_time_weeks + scenario_inv["scenario_safety_stock"]
        scenario_inv["scenario_shortage_units"] = (scenario_inv["scenario_reorder_point"] - scenario_inv["available_inventory_units"]).clip(lower=0)
        scenario_inv["scenario_stockout_risk"] = np.where(scenario_inv["scenario_shortage_units"] > 0, "High", "Low")
        scenario_inv["scenario_recommended_order_units"] = np.ceil(scenario_inv["recommended_order_units"] + scenario_inv["scenario_shortage_units"]).astype(int)

        total_shipping_cost = network["estimated_shipping_cost"].sum() * scenario["shipping_cost_multiplier"]
        weekly_capacity = warehouses["weekly_capacity_units"].sum() * scenario["capacity_multiplier"]
        future_weeks = network["week_start"].nunique()
        assigned_units = network["assigned_units"].sum() * scenario["demand_multiplier"]
        capacity_gap_units = max(0.0, assigned_units - weekly_capacity * future_weeks)

        results.append({
            "scenario_name": scenario["scenario_name"],
            "demand_multiplier": scenario["demand_multiplier"],
            "lead_time_days_added": scenario["lead_time_days_added"],
            "service_level_override": service_label,
            "shipping_cost_multiplier": scenario["shipping_cost_multiplier"],
            "capacity_multiplier": scenario["capacity_multiplier"],
            "high_stockout_risk_pairs": int((scenario_inv["scenario_stockout_risk"] == "High").sum()),
            "total_recommended_order_units": round(float(scenario_inv["scenario_recommended_order_units"].sum()), 0),
            "estimated_shipping_cost": round(float(total_shipping_cost), 2),
            "capacity_gap_units": round(float(capacity_gap_units), 2),
            "avg_stock_coverage_weeks": round(float(scenario_inv["stock_coverage_weeks"].replace(999, np.nan).mean()), 2),
        })

        detail_cols = [
            "scenario_name", "sku_id", "warehouse_id", "scenario_avg_weekly_forecast",
            "available_inventory_units", "scenario_safety_stock", "scenario_reorder_point",
            "scenario_shortage_units", "scenario_stockout_risk", "scenario_recommended_order_units",
        ]
        details.append(scenario_inv[detail_cols])

    scenario_summary = pd.DataFrame(results)
    scenario_detail = pd.concat(details, ignore_index=True)
    num_cols = scenario_detail.select_dtypes(include=[np.number]).columns
    scenario_detail[num_cols] = scenario_detail[num_cols].round(2)

    scenario_summary.to_csv(processed_dir / "scenario_results.csv", index=False)
    scenario_detail.to_csv(processed_dir / "scenario_detail.csv", index=False)
    print(f"  Scenario simulation completed. Scenarios: {len(scenario_summary)}.")

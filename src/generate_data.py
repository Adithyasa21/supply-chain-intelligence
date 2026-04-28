from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .utils import load_config, ensure_dirs


def generate_all_raw_data(root: Path) -> None:
    """Generate realistic synthetic supply-chain data.

    The data models a technology distributor with SKUs, regions, warehouses,
    orders, shipments, supplier lead times, and inventory snapshots.
    """
    cfg = load_config(root)
    ensure_dirs(root)

    rng = np.random.default_rng(int(cfg["random_seed"]))
    raw_dir = root / "data" / "raw"

    n_skus = int(cfg["n_skus"])
    n_regions = int(cfg["n_regions"])
    n_warehouses = int(cfg["n_warehouses"])
    history_weeks = int(cfg["history_weeks"])

    categories = ["Laptops", "Networking", "Accessories", "Storage", "Components", "Monitors"]
    velocity_classes = ["A", "B", "C"]

    product_rows = []
    for i in range(1, n_skus + 1):
        category = categories[(i - 1) % len(categories)]
        velocity = rng.choice(velocity_classes, p=[0.25, 0.35, 0.40])
        base_demand = {"A": rng.uniform(65, 105), "B": rng.uniform(25, 55), "C": rng.uniform(6, 22)}[velocity]
        unit_cost = float(rng.uniform(30, 900))
        margin = float(rng.uniform(0.12, 0.38))
        product_rows.append({
            "sku_id": f"SKU-{i:03d}",
            "product_name": f"{category} Product {i:03d}",
            "category": category,
            "velocity_class": velocity,
            "base_weekly_demand": round(base_demand, 2),
            "unit_cost": round(unit_cost, 2),
            "unit_price": round(unit_cost * (1 + margin), 2),
            "order_cost": round(float(rng.uniform(35, 140)), 2),
            "holding_cost_pct": round(float(rng.uniform(0.15, 0.30)), 3),
            "target_service_level": float(rng.choice([0.90, 0.92, 0.95, 0.97], p=[0.2, 0.3, 0.4, 0.1])),
        })
    products = pd.DataFrame(product_rows)

    regions = pd.DataFrame([
        {"region_id": "R-WEST", "region_name": "West", "market_size_index": 1.35, "avg_weather_delay_risk": 0.08},
        {"region_id": "R-CENTRAL", "region_name": "Central", "market_size_index": 1.05, "avg_weather_delay_risk": 0.12},
        {"region_id": "R-SOUTH", "region_name": "South", "market_size_index": 1.20, "avg_weather_delay_risk": 0.16},
        {"region_id": "R-NORTHEAST", "region_name": "Northeast", "market_size_index": 0.95, "avg_weather_delay_risk": 0.18},
        {"region_id": "R-NORTHWEST", "region_name": "Northwest", "market_size_index": 0.75, "avg_weather_delay_risk": 0.20},
    ]).head(n_regions)

    warehouses = pd.DataFrame([
        {"warehouse_id": "WH-LA", "warehouse_name": "Los Angeles DC", "region_id": "R-WEST", "weekly_capacity_units": 10500, "fixed_weekly_cost": 42000},
        {"warehouse_id": "WH-DAL", "warehouse_name": "Dallas DC", "region_id": "R-CENTRAL", "weekly_capacity_units": 12500, "fixed_weekly_cost": 46000},
        {"warehouse_id": "WH-ATL", "warehouse_name": "Atlanta DC", "region_id": "R-SOUTH", "weekly_capacity_units": 11800, "fixed_weekly_cost": 44000},
        {"warehouse_id": "WH-NJ", "warehouse_name": "New Jersey DC", "region_id": "R-NORTHEAST", "weekly_capacity_units": 10200, "fixed_weekly_cost": 48000},
    ]).head(n_warehouses)

    # Warehouse -> region cost/lead-time matrix.
    ship_rows = []
    for _, wh in warehouses.iterrows():
        for _, r in regions.iterrows():
            same_region = wh["region_id"] == r["region_id"]
            cost = rng.uniform(2.1, 4.2) if same_region else rng.uniform(4.5, 10.0)
            transit_days = int(rng.integers(1, 3)) if same_region else int(rng.integers(3, 7))
            ship_rows.append({
                "warehouse_id": wh["warehouse_id"],
                "region_id": r["region_id"],
                "shipping_cost_per_unit": round(float(cost), 2),
                "base_transit_days": transit_days,
            })
    shipping_cost_matrix = pd.DataFrame(ship_rows)

    # Supplier lead times by SKU and warehouse.
    lead_rows = []
    for _, p in products.iterrows():
        for _, wh in warehouses.iterrows():
            velocity_adjustment = {"A": -1, "B": 1, "C": 3}[p["velocity_class"]]
            avg_lead = max(3, int(rng.normal(9 + velocity_adjustment, 2)))
            lead_rows.append({
                "sku_id": p["sku_id"],
                "warehouse_id": wh["warehouse_id"],
                "primary_supplier_id": f"SUP-{rng.integers(1, 9):02d}",
                "avg_supplier_lead_time_days": avg_lead,
                "lead_time_std_days": round(float(rng.uniform(1.0, 4.0)), 2),
            })
    supplier_lead_times = pd.DataFrame(lead_rows)

    week_starts = pd.date_range(end=pd.Timestamp.today().normalize(), periods=history_weeks, freq="W-MON")
    order_rows = []
    shipment_rows = []
    order_num = 1

    # Pre-compute cheapest warehouse by region for normal fulfillment.
    cheapest = (
        shipping_cost_matrix.sort_values(["region_id", "shipping_cost_per_unit"])
        .groupby("region_id")
        .head(2)
    )

    for week_ix, week_start in enumerate(week_starts):
        # seasonality: Q4/electronics holiday demand + small sinusoidal seasonality
        month = week_start.month
        seasonal = 1.0 + 0.12 * np.sin(2 * np.pi * week_ix / 26)
        if month in [11, 12]:
            seasonal += 0.18
        if month in [1, 2]:
            seasonal -= 0.07

        for _, p in products.iterrows():
            for _, r in regions.iterrows():
                promo = int(rng.random() < (0.08 if p["velocity_class"] == "A" else 0.04))
                weather_risk = float(np.clip(r["avg_weather_delay_risk"] + rng.normal(0, 0.04), 0, 0.6))
                demand_mean = (
                    p["base_weekly_demand"]
                    * r["market_size_index"]
                    * seasonal
                    * (1.22 if promo else 1.0)
                    * float(rng.uniform(0.85, 1.15))
                )
                units = int(max(0, rng.poisson(max(1, demand_mean))))
                if units == 0:
                    continue

                region_cheapest = cheapest[cheapest["region_id"] == r["region_id"]].copy()
                # Mostly lowest-cost warehouse, sometimes second-best to mimic capacity/routing exceptions.
                wh_choice = region_cheapest.iloc[0 if rng.random() < 0.82 or len(region_cheapest) == 1 else 1]
                order_date = week_start + pd.Timedelta(days=int(rng.integers(0, 6)))
                revenue = units * float(p["unit_price"])
                order_id = f"ORD-{order_num:07d}"
                order_rows.append({
                    "order_id": order_id,
                    "order_date": order_date.date().isoformat(),
                    "week_start": week_start.date().isoformat(),
                    "sku_id": p["sku_id"],
                    "region_id": r["region_id"],
                    "warehouse_id": wh_choice["warehouse_id"],
                    "order_units": units,
                    "revenue": round(revenue, 2),
                    "promotion_flag": promo,
                    "weather_delay_risk": round(weather_risk, 3),
                })

                planned_days = int(wh_choice["base_transit_days"]) + int(rng.integers(0, 2))
                delay_days = int(rng.poisson(weather_risk * 4))
                actual_days = planned_days + delay_days
                late_flag = int(actual_days > planned_days + 1)
                shipment_rows.append({
                    "shipment_id": f"SHP-{order_num:07d}",
                    "order_id": order_id,
                    "ship_date": (order_date + pd.Timedelta(days=1)).date().isoformat(),
                    "promised_delivery_days": planned_days,
                    "actual_delivery_days": actual_days,
                    "late_delivery_flag": late_flag,
                    "shipping_cost": round(units * float(wh_choice["shipping_cost_per_unit"]), 2),
                    "weather_delay_days": delay_days,
                })
                order_num += 1

    orders = pd.DataFrame(order_rows)
    shipments = pd.DataFrame(shipment_rows)

    # Inventory snapshot derived from recent demand, with intentional over/understock variation.
    recent = (
        orders[orders["week_start"] >= str(week_starts[-12].date())]
        .groupby(["sku_id", "warehouse_id"], as_index=False)["order_units"].sum()
    )
    recent["avg_weekly_units"] = recent["order_units"] / 12

    inv_rows = []
    for _, p in products.iterrows():
        for _, wh in warehouses.iterrows():
            avg_weekly = recent[
                (recent["sku_id"] == p["sku_id"]) & (recent["warehouse_id"] == wh["warehouse_id"])
            ]["avg_weekly_units"]
            avg_weekly = float(avg_weekly.iloc[0]) if len(avg_weekly) else float(p["base_weekly_demand"] * 0.15)
            stock_weeks = float(rng.choice([0.5, 1.0, 2.0, 4.0, 8.0, 12.0], p=[0.10, 0.15, 0.25, 0.25, 0.18, 0.07]))
            current = int(max(0, avg_weekly * stock_weeks + rng.normal(0, max(1, avg_weekly * 0.5))))
            reserved = int(max(0, current * rng.uniform(0.02, 0.18)))
            inv_rows.append({
                "sku_id": p["sku_id"],
                "warehouse_id": wh["warehouse_id"],
                "current_inventory_units": current,
                "reserved_inventory_units": reserved,
            })
    inventory_snapshot = pd.DataFrame(inv_rows)

    for name, df in {
        "products": products,
        "regions": regions,
        "warehouses": warehouses,
        "shipping_cost_matrix": shipping_cost_matrix,
        "supplier_lead_times": supplier_lead_times,
        "orders": orders,
        "shipments": shipments,
        "inventory_snapshot": inventory_snapshot,
    }.items():
        df.to_csv(raw_dir / f"{name}.csv", index=False)

    print(f"  Generated {len(orders):,} orders, {len(products)} SKUs, {len(warehouses)} warehouses, {len(regions)} regions.")

-- Processed-layer SQL views created after the pipeline writes forecast/optimization tables.

DROP VIEW IF EXISTS v_powerbi_inventory_risk;
CREATE VIEW v_powerbi_inventory_risk AS
SELECT
    sku_id,
    product_name,
    warehouse_id,
    category,
    velocity_class,
    avg_weekly_forecast_units,
    available_inventory_units,
    safety_stock_units,
    reorder_point_units,
    eoq_units,
    stock_coverage_weeks,
    stockout_risk,
    overstock_risk,
    recommended_order_units,
    estimated_replenishment_cost
FROM inventory_recommendations;

DROP VIEW IF EXISTS v_powerbi_scenario_comparison;
CREATE VIEW v_powerbi_scenario_comparison AS
SELECT
    scenario_name,
    demand_multiplier,
    lead_time_days_added,
    service_level_override,
    high_stockout_risk_pairs,
    total_recommended_order_units,
    estimated_shipping_cost,
    capacity_gap_units
FROM scenario_results;

DROP VIEW IF EXISTS v_powerbi_network_summary;
CREATE VIEW v_powerbi_network_summary AS
SELECT
    week_start,
    warehouse_id,
    region_id,
    SUM(assigned_units) AS assigned_units,
    ROUND(SUM(estimated_shipping_cost), 2) AS estimated_shipping_cost,
    SUM(unmet_units) AS unmet_units
FROM network_recommendations
GROUP BY week_start, warehouse_id, region_id;

-- Useful KPI queries for interviews / Power BI / SQL demonstrations.

-- 1) Monthly revenue, units, and late delivery rate
SELECT
    substr(o.order_date, 1, 7) AS month,
    COUNT(DISTINCT o.order_id) AS orders,
    SUM(o.order_units) AS units,
    ROUND(SUM(o.revenue), 2) AS revenue,
    ROUND(AVG(s.late_delivery_flag) * 100, 2) AS late_delivery_rate_pct
FROM orders o
LEFT JOIN shipments s ON o.order_id = s.order_id
GROUP BY substr(o.order_date, 1, 7)
ORDER BY month;

-- 2) Top SKU-warehouse stockout risks
SELECT
    sku_id,
    product_name,
    warehouse_id,
    category,
    avg_weekly_forecast_units,
    available_inventory_units,
    reorder_point_units,
    shortage_units,
    recommended_order_units,
    estimated_replenishment_cost
FROM inventory_recommendations
WHERE stockout_risk = 'High'
ORDER BY shortage_units DESC
LIMIT 20;

-- 3) Overstock candidates
SELECT
    sku_id,
    product_name,
    warehouse_id,
    category,
    excess_units,
    stock_coverage_weeks,
    estimated_holding_cost_annual
FROM inventory_recommendations
WHERE overstock_risk = 'High'
ORDER BY excess_units DESC
LIMIT 20;

-- 4) Scenario comparison
SELECT
    scenario_name,
    high_stockout_risk_pairs,
    total_recommended_order_units,
    estimated_shipping_cost,
    capacity_gap_units
FROM scenario_results
ORDER BY high_stockout_risk_pairs DESC;

-- 5) Warehouse-to-region allocation cost
SELECT
    warehouse_id,
    region_id,
    SUM(assigned_units) AS assigned_units,
    ROUND(SUM(estimated_shipping_cost), 2) AS estimated_shipping_cost
FROM network_recommendations
GROUP BY warehouse_id, region_id
ORDER BY estimated_shipping_cost DESC;

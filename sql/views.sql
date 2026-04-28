-- Raw-layer SQL views created early in the pipeline.

DROP VIEW IF EXISTS v_order_service_kpis;
CREATE VIEW v_order_service_kpis AS
SELECT
    o.week_start,
    o.region_id,
    o.warehouse_id,
    p.category,
    COUNT(DISTINCT o.order_id) AS order_count,
    SUM(o.order_units) AS units,
    ROUND(SUM(o.revenue), 2) AS revenue,
    ROUND(SUM(s.shipping_cost), 2) AS shipping_cost,
    ROUND(AVG(s.late_delivery_flag) * 100, 2) AS late_delivery_rate_pct
FROM orders o
JOIN products p ON o.sku_id = p.sku_id
LEFT JOIN shipments s ON o.order_id = s.order_id
GROUP BY o.week_start, o.region_id, o.warehouse_id, p.category;

DROP VIEW IF EXISTS v_current_inventory_position;
CREATE VIEW v_current_inventory_position AS
SELECT
    i.sku_id,
    p.product_name,
    p.category,
    p.velocity_class,
    i.warehouse_id,
    w.warehouse_name,
    i.current_inventory_units,
    i.reserved_inventory_units,
    (i.current_inventory_units - i.reserved_inventory_units) AS available_inventory_units,
    p.unit_cost,
    ROUND((i.current_inventory_units - i.reserved_inventory_units) * p.unit_cost, 2) AS inventory_value
FROM inventory_snapshot i
JOIN products p ON i.sku_id = p.sku_id
JOIN warehouses w ON i.warehouse_id = w.warehouse_id;

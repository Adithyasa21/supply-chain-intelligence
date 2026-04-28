-- Azure-Ready Supply Chain Control Tower
-- SQLite-compatible schema. The same logical model can be recreated in Azure SQL.

CREATE TABLE IF NOT EXISTS products (
    sku_id TEXT PRIMARY KEY,
    product_name TEXT,
    category TEXT,
    velocity_class TEXT,
    base_weekly_demand REAL,
    unit_cost REAL,
    unit_price REAL,
    order_cost REAL,
    holding_cost_pct REAL,
    target_service_level REAL
);

CREATE TABLE IF NOT EXISTS regions (
    region_id TEXT PRIMARY KEY,
    region_name TEXT,
    market_size_index REAL,
    avg_weather_delay_risk REAL
);

CREATE TABLE IF NOT EXISTS warehouses (
    warehouse_id TEXT PRIMARY KEY,
    warehouse_name TEXT,
    region_id TEXT,
    weekly_capacity_units INTEGER,
    fixed_weekly_cost REAL
);

CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    order_date TEXT,
    week_start TEXT,
    sku_id TEXT,
    region_id TEXT,
    warehouse_id TEXT,
    order_units INTEGER,
    revenue REAL,
    promotion_flag INTEGER,
    weather_delay_risk REAL
);

CREATE TABLE IF NOT EXISTS shipments (
    shipment_id TEXT PRIMARY KEY,
    order_id TEXT,
    ship_date TEXT,
    promised_delivery_days INTEGER,
    actual_delivery_days INTEGER,
    late_delivery_flag INTEGER,
    shipping_cost REAL,
    weather_delay_days INTEGER
);

CREATE TABLE IF NOT EXISTS inventory_snapshot (
    sku_id TEXT,
    warehouse_id TEXT,
    current_inventory_units INTEGER,
    reserved_inventory_units INTEGER
);

CREATE TABLE IF NOT EXISTS supplier_lead_times (
    sku_id TEXT,
    warehouse_id TEXT,
    primary_supplier_id TEXT,
    avg_supplier_lead_time_days INTEGER,
    lead_time_std_days REAL
);

CREATE TABLE IF NOT EXISTS shipping_cost_matrix (
    warehouse_id TEXT,
    region_id TEXT,
    shipping_cost_per_unit REAL,
    base_transit_days INTEGER
);

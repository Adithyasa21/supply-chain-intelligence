# Project Report

## Problem

Supply-chain teams often face three connected problems:

1. Demand uncertainty causes overstocking or stockouts.
2. Inventory policies are not always linked to updated demand forecasts.
3. Leaders need scenario-based views of cost, service level, and fulfillment risk.

## Objective

Build a control tower that forecasts future demand, recommends inventory actions, simulates disruptions, and provides Power BI-ready outputs for planning teams.

## Data

Synthetic data includes:

- Products/SKUs
- Regions
- Warehouses
- Orders
- Shipments
- Inventory snapshots
- Supplier lead times
- Shipping cost matrix

## Methods

### Demand forecasting

A Random Forest model forecasts SKU-region demand using lag features, rolling averages, promotion rate, weather-delay risk, region market size, product category, and velocity class.

### Inventory optimization

The project calculates:

- Safety stock
- Reorder point
- EOQ
- Stock coverage weeks
- Inventory turnover
- Stockout risk
- Overstock risk
- Recommended order units

### Scenario simulation

Scenarios include:

- Demand surge +20%
- Supplier delay +5 days
- Higher service level 97%
- Shipping cost +10%
- Warehouse capacity -15%

### Network allocation

The system assigns forecasted demand to warehouses using a cost-aware greedy model with weekly warehouse capacity.

## Business outputs

The control tower produces:

- High-risk SKU list
- Overstock candidate list
- Replenishment cost estimate
- Scenario comparison
- Forecast accuracy table
- Warehouse-to-region allocation view
- Executive KPI summary

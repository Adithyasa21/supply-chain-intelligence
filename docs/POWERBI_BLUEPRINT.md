# Power BI Dashboard Blueprint

Use the CSVs in `data/powerbi_outputs/`.

## Page 1: Executive Overview

Data source:
- `executive_summary.csv`
- `monthly_kpis.csv`

Cards:
- Total Orders
- Total Revenue
- Forecast MAPE %
- Stockout-Risk SKU-Warehouse Pairs
- Estimated Replenishment Cost
- Late Delivery Rate %

Charts:
- Monthly units and orders trend
- Monthly revenue
- Late-delivery rate trend

## Page 2: Demand Forecasting

Data source:
- `forecast_accuracy_detail.csv`
- `forecast_feature_importance_top30.csv`

Charts:
- Actual vs forecasted demand
- Forecast error by SKU
- Top model features
- Forecast error by region

## Page 3: Inventory Risk

Data source:
- `inventory_recommendations.csv`
- `category_inventory_risk.csv`

Charts:
- Stockout vs overstock risk by category
- Recommended order units by warehouse
- Replenishment cost by category
- Table of high-risk SKUs

## Page 4: Network Allocation

Data source:
- `network_recommendations.csv`

Charts:
- Assigned units by warehouse and region
- Estimated shipping cost by warehouse-region lane
- Unmet demand by week

## Page 5: Scenario Analysis

Data source:
- `scenario_results.csv`
- `scenario_detail.csv`

Charts:
- Stockout-risk pairs by scenario
- Recommended order units by scenario
- Shipping cost by scenario
- Capacity gap by scenario

## Design style

Use a clean executive theme:
- white or very light background
- dark navy headers
- muted accent colors
- one business question per page
- keep the recommendation table visible

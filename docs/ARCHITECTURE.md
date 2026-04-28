# Architecture

## Local standalone architecture

```text
1. Synthetic raw data
   products, regions, warehouses, orders, shipments, inventory, suppliers

2. Python ETL and ML
   generate_data.py → forecasting.py → inventory_optimization.py

3. SQL layer
   SQLite database with raw and processed tables
   SQL scripts can be converted to Azure SQL

4. Scenario and network layer
   scenario_simulation.py and network_optimization.py

5. Reporting layer
   CSV outputs for Power BI and a Streamlit app for local demo
```

## Azure-lite architecture

```text
Azure Blob Storage / Data Lake
        ↓
Python scheduled job / Azure ML notebook
        ↓
Azure SQL Database
        ↓
Power BI semantic model
        ↓
Power BI executive dashboard
```

## Why this is better than a normal dashboard project

A dashboard alone only visualizes what happened. This project also recommends what to do:

- reorder SKUs below reorder point
- identify overstocked inventory
- simulate demand or supplier disruptions
- quantify cost-service tradeoffs
- allocate demand to lower-cost warehouses under capacity limits

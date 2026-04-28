# Interview Script

## 30-second explanation

I built an Azure-ready supply-chain forecasting and inventory optimization control tower. The project simulates a technology distributor with SKUs, warehouses, regions, orders, shipments, inventory, and supplier lead times. I used Python to forecast SKU-region demand, translated the forecast into safety stock, reorder point, EOQ, and stockout-risk recommendations, then created scenario simulations for demand surges, supplier delays, service-level changes, shipping-cost increases, and warehouse capacity reductions. The output is stored in a SQL-ready model and exported into Power BI-ready tables for executive decision-making.

## Business explanation

The goal was to move from descriptive reporting to planning recommendations. Instead of only showing historical orders, the system answers: which products will stock out, which products are overstocked, what should be reordered, what happens if lead time increases, and how much cost or risk changes under different scenarios.

## Technical explanation

I generated a relational supply-chain dataset, built a SQLite database that can map to Azure SQL, engineered lag and rolling-demand features, trained a Random Forest forecasting model, calculated inventory policies using service-level-based safety stock, reorder point, EOQ, and stock coverage days, and used a greedy network-allocation model to assign forecasted demand to lower-cost warehouses under capacity constraints.

## Why Azure-ready

The project runs locally so it is easy to reproduce, but the architecture maps directly to Azure: raw files can land in Azure Blob Storage, Python jobs can run in Azure ML or scheduled compute, processed tables can go into Azure SQL, and Power BI can connect to the SQL model.

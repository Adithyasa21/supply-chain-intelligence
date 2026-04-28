# Azure-Lite Deployment Plan

You do not need to deploy the full Microsoft accelerator to get resume value. Use this mapping:

| Local component | Azure equivalent |
|---|---|
| `data/raw/*.csv` | Azure Blob Storage / Data Lake Storage |
| Python scripts | Azure ML notebook, Azure Function, or scheduled job |
| `supply_chain_control_tower.db` | Azure SQL Database |
| `data/powerbi_outputs/*.csv` | Power BI dataset / Dataflow |
| `app.py` Streamlit app | Optional Azure App Service |
| `sql/*.sql` | Azure SQL schema and views |

## Suggested GitHub wording

This project runs locally for reproducibility and is designed with an Azure-ready architecture. In production, raw supply-chain files would land in Azure Blob Storage, Python forecasting/optimization jobs would run through Azure ML or scheduled compute, processed outputs would be persisted to Azure SQL Database, and Power BI would serve the control-tower dashboard to business users.

## What not to overbuild

Do not waste time on:

- Kubernetes
- Ray distributed execution
- Power Apps
- enterprise authentication
- production CI/CD

Those are valuable in enterprise systems, but for a portfolio project the strongest signal is the business analytics flow: data → forecast → inventory policy → scenarios → dashboard → recommendations.

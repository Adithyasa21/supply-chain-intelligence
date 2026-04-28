# Notes on Uploaded Microsoft Accelerator

The uploaded Microsoft Intelligent Supply Chain Management Accelerator was used as architecture inspiration, not copied as a deployed cloud project.

## What was borrowed conceptually

- Control-tower thinking
- Forecasting → inventory optimization → simulation → Power BI reporting flow
- Azure-ready architecture pattern
- Business-user reporting orientation

## What was intentionally simplified

The original accelerator references Azure Data Factory, Data Lake, Azure SQL, Azure ML, Kubernetes, Ray, Power Apps, and Power BI. For a resume-ready standalone project, this package keeps the high-value analytics parts and removes heavyweight deployment requirements.

## Why this approach is better for a portfolio

A recruiter or hiring manager can run the project locally, inspect the SQL/Python logic, and understand the business recommendations. You can still explain how the project would be deployed on Azure in production.

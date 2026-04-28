"""
One-command runner for the Supply Chain Intelligence Platform.

Usage:
    python run_pipeline.py               # local mode
    python run_pipeline.py --azure       # upload all outputs to Azure

Azure mode requires AZURE_STORAGE_ACCOUNT_NAME in environment / .env file.
Run infra/provision.sh once to create all Azure resources.

Pipeline stages
---------------
 1. Generate synthetic supply-chain data
 2. Build SQLite database + SQL views
 3. Train demand forecasting model (Random Forest)  [MLflow tracked]
 4. Compute conformal prediction intervals (Mondrian)
 5. Run EOQ-based inventory optimization
 6. Solve multi-period lot-sizing MIP (PuLP/CBC)
 7. Greedy warehouse-to-region allocation
 8. Run supply chain resilience analysis (NetworkX)
 9. Monte Carlo CVaR risk engine (10,000 scenarios)
10. Scenario simulation + Power BI outputs
── Azure steps (when --azure or AZURE_STORAGE_ACCOUNT_NAME is set) ──
11. Upload Bronze / Silver / Gold layers to ADLS Gen2
12. Log all metrics + model artifact to MLflow / Azure ML
"""
import argparse
import sys
from pathlib import Path

from src.generate_data import generate_all_raw_data
from src.database import build_sqlite_database
from src.forecasting import run_forecasting_pipeline
from src.conformal_forecasting import run_conformal_forecasting
from src.inventory_optimization import run_inventory_optimization
from src.stochastic_optimizer import run_stochastic_optimizer
from src.network_optimization import run_network_optimization
from src.resilience import run_resilience_analysis
from src.risk_engine import run_risk_engine
from src.scenario_simulation import run_scenario_simulation
from src.reporting import build_powerbi_outputs


def _parse_args():
    p = argparse.ArgumentParser(description="Supply Chain Intelligence Pipeline")
    p.add_argument(
        "--azure", action="store_true",
        help="Upload outputs to Azure ADLS Gen2 and log to Azure ML / MLflow",
    )
    p.add_argument(
        "--skip-data-gen", action="store_true",
        help="Skip step 1 (reuse existing raw data)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    root = Path(__file__).resolve().parent

    # ── Azure setup (non-blocking if not configured) ─────────────────────────
    from cloud.config import azure_config
    from cloud.storage import DataLakeStore
    from cloud.ml_tracking import MLTracker

    use_azure = args.azure or azure_config.enabled
    store: DataLakeStore | None = DataLakeStore.from_config(azure_config) if use_azure else None
    tracker = MLTracker.from_config(azure_config)

    if use_azure and not store:
        print("⚠  Azure storage not reachable — running in local mode.")
    if use_azure and store:
        print(f"✓  Azure ADLS Gen2 connected: {azure_config.storage_account_name}/{azure_config.storage_filesystem}")
    print(f"✓  MLflow tracking: {'Azure ML' if tracker._azure_mode else 'local (./mlruns)'}")

    print("\n=== Supply Chain Intelligence Platform ===")

    with tracker.start_run("full-pipeline"):
        tracker.set_tags({
            "pipeline_version": "2.0.0",
            "azure_mode": str(use_azure),
            "storage_account": azure_config.storage_account_name or "local",
        })

        # ── Step 1: Data generation ──────────────────────────────────────────
        if not args.skip_data_gen:
            print("Step  1/10: Generating synthetic supply-chain data...")
            generate_all_raw_data(root)
        else:
            print("Step  1/10: Skipped (--skip-data-gen).")

        # ── Step 2: Database ─────────────────────────────────────────────────
        print("Step  2/10: Building SQLite database and SQL views...")
        build_sqlite_database(root)

        # ── Step 3: Forecasting + MLflow tracking ────────────────────────────
        print("Step  3/10: Training demand forecast model (Random Forest)...")
        run_forecasting_pipeline(root)

        # Log forecast metrics and model to MLflow
        import pandas as pd
        from joblib import load as jl_load

        metrics_path = root / "data" / "processed" / "forecast_metrics.csv"
        if metrics_path.exists():
            m = pd.read_csv(metrics_path).iloc[0]
            tracker.log_params({
                "model_type": "RandomForestRegressor",
                "n_estimators": 80,
                "max_depth": 14,
                "min_samples_leaf": 3,
                "forecast_horizon_weeks": 8,
            })
            tracker.log_metrics({
                "forecast_mape_pct": float(m["mape_pct"]),
                "forecast_rmse": float(m["rmse"]),
                "forecast_bias_pct": float(m["forecast_bias_pct"]),
            })

        model_path = root / "models" / "demand_forecast_random_forest.pkl"
        if model_path.exists():
            model = jl_load(model_path)
            tracker.log_model(model, artifact_path="demand_forecast")

        fi_path = root / "data" / "processed" / "forecast_feature_importance.csv"
        if fi_path.exists():
            tracker.log_feature_importance(pd.read_csv(fi_path))

        # ── Step 4: Conformal prediction ─────────────────────────────────────
        print("Step  4/10: Computing conformal prediction intervals (Mondrian)...")
        run_conformal_forecasting(root)

        cal_path = root / "data" / "processed" / "conformal_calibration_metrics.csv"
        if cal_path.exists():
            tracker.log_conformal_calibration(pd.read_csv(cal_path))

        # ── Step 5: EOQ inventory ────────────────────────────────────────────
        print("Step  5/10: Calculating EOQ inventory policies...")
        run_inventory_optimization(root)

        # ── Step 6: MIP optimizer ────────────────────────────────────────────
        print("Step  6/10: Solving multi-period lot-sizing MIP (PuLP/CBC)...")
        run_stochastic_optimizer(root)

        mip_path = root / "data" / "processed" / "mip_inventory_summary.csv"
        if mip_path.exists():
            tracker.log_mip_summary(pd.read_csv(mip_path))

        # ── Step 7: Network allocation ───────────────────────────────────────
        print("Step  7/10: Optimizing warehouse-to-region network allocation...")
        run_network_optimization(root)

        # ── Step 8: Resilience ───────────────────────────────────────────────
        print("Step  8/10: Running supply chain resilience analysis (NetworkX)...")
        run_resilience_analysis(root)

        res_path = root / "data" / "processed" / "resilience_scores.csv"
        if res_path.exists():
            tracker.log_resilience_summary(pd.read_csv(res_path))

        # ── Step 9: Risk engine ──────────────────────────────────────────────
        print("Step  9/10: Running Monte Carlo CVaR risk engine (10,000 scenarios)...")
        run_risk_engine(root)

        risk_path = root / "data" / "processed" / "risk_metrics.csv"
        if risk_path.exists():
            tracker.log_risk_summary(pd.read_csv(risk_path))

        # ── Step 10: Scenario simulation + reporting ─────────────────────────
        print("Step 10/10: Scenario simulation + Power BI outputs...")
        run_scenario_simulation(root)
        build_powerbi_outputs(root)

        # ── Azure: upload all outputs to ADLS Gen2 ───────────────────────────
        if store:
            print("\nAzure Step 11: Uploading outputs to ADLS Gen2 medallion layers...")
            uploaded = store.upload_all_outputs(root)
            n_bronze = len(uploaded["bronze"])
            n_silver = len(uploaded["silver"])
            n_gold = len(uploaded["gold"])
            print(f"  ✓ Bronze: {n_bronze} files | Silver: {n_silver} files | Gold: {n_gold} files")

            # Write run manifest to gold layer for lineage/audit
            store.write_pipeline_manifest({
                "mlflow_run_id": tracker.run_id,
                "bronze_files": n_bronze,
                "silver_files": n_silver,
                "gold_files": n_gold,
            })
            print(f"  ✓ Pipeline manifest written to gold/serving/supply_chain/manifests/")

        # ── Promote model to Staging in Azure ML registry ────────────────────
        if tracker._azure_mode:
            tracker.promote_model_to_staging()
            print("  ✓ Model promoted to Staging in Azure ML Model Registry")

    print("\nPipeline completed successfully.")
    print(f"\nMLflow run ID: {tracker.run_id or 'see ./mlruns/'}")
    print("\nKey outputs:")
    print("  data/processed/forecast_intervals.csv       — conformal prediction intervals")
    print("  data/processed/mip_order_schedule.csv       — MIP optimal ordering schedule")
    print("  data/processed/resilience_scores.csv        — network resilience + SPOF flags")
    print("  data/processed/risk_metrics.csv             — VaR / CVaR per SKU-warehouse")
    if store:
        print(f"\n  Azure ADLS Gen2: https://{azure_config.storage_account_name}.dfs.core.windows.net/{azure_config.storage_filesystem}/")
    print("\nStart dashboard:   streamlit run app.py")
    print("Start REST API:    uvicorn api.main:app --reload --port 8000")
    print("View MLflow runs:  mlflow ui   (or https://ml.azure.com in Azure mode)")


if __name__ == "__main__":
    main()

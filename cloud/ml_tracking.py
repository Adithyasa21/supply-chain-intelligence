"""
Azure ML Experiment Tracking via MLflow.

Azure ML natively uses MLflow as its tracking protocol.
The workspace exposes an MLflow tracking URI — point mlflow at it and
every run, metric, parameter, and artifact is stored in Azure ML with
full lineage, comparison UI, and model registry integration.

Why this matters
----------------
Without experiment tracking:
  - You can't compare "did last week's model do better?"
  - You can't reproduce a model (which data? which hyperparameters?)
  - You can't know which deployed model produced which forecast

With Azure ML + MLflow:
  - Every training run is logged (params, metrics, artifacts)
  - Models are versioned and staged (Staging → Production → Archived)
  - You can trigger retraining pipelines from the registry
  - Full audit trail for compliance

Architecture used here
----------------------
Local mode:  MLflow logs to ./mlruns/ (SQLite + local filesystem)
Azure mode:  MLflow tracking URI pointed at Azure ML workspace
             → runs appear in Azure ML Studio UI
             → models go into the Azure ML Model Registry

Model registry transitions
--------------------------
  None → Staging  (after training, automatically)
  Staging → Production  (after manual review or evaluation gate)
  Production → Archived  (when a better model is promoted)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import mlflow
import mlflow.sklearn

logger = logging.getLogger(__name__)

_EXPERIMENT_NAME = "supply-chain-demand-forecasting"
_MODEL_NAME = "demand-forecast-rf"


class MLTracker:
    """
    Wraps MLflow with Azure ML backend when configured, local otherwise.

    Usage
    -----
    tracker = MLTracker.from_config(azure_config)
    with tracker.start_run("training") as run:
        tracker.log_params({"n_estimators": 80, "max_depth": 14})
        tracker.log_metrics({"mape": 22.93, "rmse": 8.12, "bias": 0.08})
        tracker.log_model(sklearn_model, "random_forest")
        tracker.log_feature_importance(fi_df)
        run_id = tracker.run_id
    """

    def __init__(self, tracking_uri: Optional[str] = None) -> None:
        self._azure_mode = tracking_uri is not None
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            logger.info("MLflow → Azure ML: %s", tracking_uri)
        else:
            # Local: log to ./mlruns relative to project root
            local_uri = str(Path(__file__).resolve().parent.parent / "mlruns")
            mlflow.set_tracking_uri(f"file://{local_uri}")
            logger.info("MLflow → local: %s", local_uri)

        mlflow.set_experiment(_EXPERIMENT_NAME)
        self._run: Optional[mlflow.ActiveRun] = None

    # ── Factory ───────────────────────────────────────────────────────────────
    @classmethod
    def from_config(cls, cfg) -> "MLTracker":
        """
        Build tracker pointing at Azure ML if configured, else local MLflow.

        Azure ML MLflow tracking URI format:
          azureml://subscriptions/<sub>/resourceGroups/<rg>/
                   providers/Microsoft.MachineLearningServices/workspaces/<ws>
        """
        tracking_uri: Optional[str] = None

        if cfg.ml_configured:
            try:
                from azure.identity import DefaultAzureCredential
                from azure.ai.ml import MLClient

                credential = DefaultAzureCredential()
                ml_client = MLClient(
                    credential=credential,
                    subscription_id=cfg.ml_subscription_id,
                    resource_group_name=cfg.ml_resource_group,
                    workspace_name=cfg.ml_workspace_name,
                )
                tracking_uri = ml_client.workspaces.get(cfg.ml_workspace_name).mlflow_tracking_uri
                logger.info("Azure ML workspace connected: %s", cfg.ml_workspace_name)
            except ImportError:
                # azure-ai-ml not installed — use REST-based URI directly
                tracking_uri = (
                    f"azureml://subscriptions/{cfg.ml_subscription_id}"
                    f"/resourceGroups/{cfg.ml_resource_group}"
                    f"/providers/Microsoft.MachineLearningServices"
                    f"/workspaces/{cfg.ml_workspace_name}"
                )
            except Exception as exc:
                logger.warning("Azure ML unavailable (%s). Using local MLflow.", exc)
                tracking_uri = None

        return cls(tracking_uri)

    # ── Context manager for a training run ───────────────────────────────────
    def start_run(self, run_name: str = "pipeline-run") -> "MLTracker":
        self._run = mlflow.start_run(run_name=run_name)
        return self

    def end_run(self) -> None:
        if self._run:
            mlflow.end_run()
            self._run = None

    def __enter__(self) -> "MLTracker":
        return self

    def __exit__(self, *_) -> None:
        self.end_run()

    @property
    def run_id(self) -> Optional[str]:
        active = mlflow.active_run()
        return active.info.run_id if active else None

    # ── Logging helpers ───────────────────────────────────────────────────────
    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters. Immutable after logging — enables reproducibility."""
        mlflow.log_params({str(k): str(v) for k, v in params.items()})

    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        """Log evaluation metrics. Supports step for time-series tracking."""
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)

    def log_model(self, model, artifact_path: str = "model") -> str:
        """
        Log sklearn model to MLflow.
        In Azure ML mode this registers to the Azure ML Model Registry.
        Returns the model URI.
        """
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            registered_model_name=_MODEL_NAME if self._azure_mode else None,
        )
        return model_info.model_uri

    def log_feature_importance(self, fi_df) -> None:
        """Log feature importance as a CSV artifact and top-5 as metrics."""
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            fi_df.to_csv(f, index=False)
            tmp = f.name
        mlflow.log_artifact(tmp, artifact_path="feature_importance")
        os.unlink(tmp)

        # Log top-5 features as named metrics for quick comparison in UI
        for _, row in fi_df.head(5).iterrows():
            safe_name = str(row["feature"]).replace("=", "_").replace("-", "_")[:50]
            mlflow.log_metric(f"fi_{safe_name}", float(row["importance"]))

    def log_conformal_calibration(self, cal_df) -> None:
        """Log conformal calibration metrics per velocity class."""
        for _, row in cal_df.iterrows():
            vc = str(row["velocity_class"])
            mlflow.log_metrics({
                f"conformal_coverage_{vc}": float(row["empirical_coverage"]),
                f"conformal_q_hat_{vc}": float(row["q_hat"]),
            })

    def log_mip_summary(self, summary_df) -> None:
        """Log MIP optimization aggregate metrics."""
        mlflow.log_metrics({
            "mip_total_cost": float(summary_df["mip_total_cost"].sum()),
            "eoq_benchmark_cost": float(summary_df["eoq_benchmark_cost"].sum()),
            "mip_savings_vs_eoq": float(summary_df["cost_savings_vs_eoq"].sum()),
            "mip_avg_savings_pct": float(summary_df["savings_pct"].mean()),
            "mip_optimal_pct": float((summary_df["mip_status"] == "Optimal").mean() * 100),
        })

    def log_risk_summary(self, risk_df) -> None:
        """Log Monte Carlo risk aggregate metrics."""
        mlflow.log_metrics({
            "risk_critical_pairs": int((risk_df["risk_tier"] == "Critical").sum()),
            "risk_avg_cvar_95": float(risk_df["cvar_95"].mean()),
            "risk_avg_cvar_ratio": float(risk_df["cvar_to_mean_ratio"].mean()),
            "risk_avg_stockout_prob": float(risk_df["stockout_probability"].mean()),
        })

    def log_resilience_summary(self, res_df) -> None:
        """Log resilience analysis metrics."""
        mlflow.log_metrics({
            "resilience_n_spof": int(res_df["is_single_point_of_failure"].sum()),
            "resilience_min_score": float(res_df["resilience_score"].min()),
            "resilience_avg_score": float(res_df["resilience_score"].mean()),
        })

    def set_tags(self, tags: dict[str, str]) -> None:
        mlflow.set_tags(tags)

    # ── Model registry transition (Azure ML only) ────────────────────────────
    def promote_model_to_staging(self, version: Optional[str] = None) -> None:
        """
        Transition the latest registered model version to Staging.
        In production, this would be gated by an evaluation check
        (e.g. MAPE < threshold) before promotion to Production.
        """
        if not self._azure_mode:
            logger.info("Model promotion only available in Azure ML mode.")
            return
        try:
            client = mlflow.tracking.MlflowClient()
            versions = client.get_latest_versions(_MODEL_NAME, stages=["None"])
            if versions:
                v = versions[0].version
                client.transition_model_version_stage(_MODEL_NAME, v, "Staging")
                logger.info("Model %s v%s → Staging", _MODEL_NAME, v)
        except Exception as exc:
            logger.warning("Model promotion failed: %s", exc)

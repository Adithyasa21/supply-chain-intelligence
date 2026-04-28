"""
Mondrian Split Conformal Prediction for calibrated demand forecast intervals.

Most ML forecast systems report point estimates with no coverage guarantee.
Bootstrapped or ±2σ intervals are asymptotically valid but provide no
finite-sample guarantees.

Split conformal prediction (Papadopoulos 2002; Vovk, Gammerman & Shafer 2005)
gives a provably valid marginal coverage guarantee:

    P(Y_test ∈ Ĉ(X_test)) ≥ 1 − α

with NO distributional assumptions on (X, Y) and regardless of model quality.

Algorithm
---------
1. Use the backtest holdout as the calibration set (model never saw this data).
2. Nonconformity score for each calibration point i:  s_i = |y_i − ŷ_i|
3. q̂ = Quantile(s_1…s_n, level = ⌈(n+1)(1−α)⌉ / n)
4. For any future point:  Ĉ(x) = [ŷ(x) − q̂,  ŷ(x) + q̂]

Mondrian Conformal (conditional coverage by velocity class)
------------------------------------------------------------
Separate q̂ per class (A / B / C) gives conditional coverage:

    P(Y ∈ Ĉ(X) | velocity_class = k) ≥ 1 − α

SKU-level Mondrian is also computed when ≥ 15 calibration points exist per SKU,
giving the tightest possible intervals for well-behaved SKUs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


_ALPHA = 0.10          # target miscoverage → 90 % prediction intervals
_MIN_SKU_POINTS = 15   # minimum calibration points to use SKU-level quantile


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Return the (1−α)-conformal quantile with finite-sample correction."""
    n = len(scores)
    if n == 0:
        return float("inf")
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level))


def run_conformal_forecasting(root: Path) -> None:
    processed = root / "data" / "processed"
    raw = root / "data" / "raw"

    backtest = pd.read_csv(processed / "forecast_backtest.csv")
    products = pd.read_csv(raw / "products.csv")
    forecast = pd.read_csv(processed / "forecast_output.csv")

    # Join velocity class onto calibration set
    backtest = backtest.merge(
        products[["sku_id", "velocity_class"]], on="sku_id", how="left"
    )

    # Nonconformity scores: absolute residuals
    backtest["score"] = (
        backtest["actual_demand"] - backtest["forecasted_demand"]
    ).abs()

    # ── Global (marginal) quantile ──────────────────────────────────────────
    global_scores = backtest["score"].values
    q_global = _conformal_quantile(global_scores, _ALPHA)

    # ── Mondrian quantiles per velocity class (conditional coverage) ────────
    mondrian_q: dict[str, float] = {}
    for vc, grp in backtest.groupby("velocity_class"):
        mondrian_q[str(vc)] = _conformal_quantile(grp["score"].values, _ALPHA)

    # ── SKU-level Mondrian (tightest intervals for well-observed SKUs) ───────
    sku_q: dict[str, float] = {}
    for sku, grp in backtest.groupby("sku_id"):
        if len(grp) >= _MIN_SKU_POINTS:
            sku_q[str(sku)] = _conformal_quantile(grp["score"].values, _ALPHA)

    # ── Empirical coverage on calibration set (sanity check) ────────────────
    backtest["covered_global"] = (
        (backtest["actual_demand"] >= backtest["forecasted_demand"] - q_global)
        & (backtest["actual_demand"] <= backtest["forecasted_demand"] + q_global)
    )
    for vc, grp in backtest.groupby("velocity_class"):
        q = mondrian_q[str(vc)]
        backtest.loc[grp.index, "covered_mondrian"] = (
            (grp["actual_demand"] >= grp["forecasted_demand"] - q)
            & (grp["actual_demand"] <= grp["forecasted_demand"] + q)
        )

    empirical_global = float(backtest["covered_global"].mean())
    empirical_mondrian = float(backtest["covered_mondrian"].mean())

    # ── Apply intervals to future forecast ──────────────────────────────────
    forecast = forecast.merge(
        products[["sku_id", "velocity_class"]], on="sku_id", how="left"
    )

    # Choose finest available quantile: SKU > velocity class > global
    def _choose_q(row: pd.Series) -> float:
        if row["sku_id"] in sku_q:
            return sku_q[row["sku_id"]]
        return mondrian_q.get(str(row["velocity_class"]), q_global)

    forecast["q_hat"] = forecast.apply(_choose_q, axis=1)
    forecast["lower_bound"] = (
        forecast["forecasted_demand"] - forecast["q_hat"]
    ).clip(lower=0.0)
    forecast["upper_bound"] = forecast["forecasted_demand"] + forecast["q_hat"]
    forecast["interval_width"] = forecast["upper_bound"] - forecast["lower_bound"]
    forecast["coverage_target"] = 1.0 - _ALPHA

    # Velocity-class q_hat for display
    forecast["velocity_q_hat"] = forecast["velocity_class"].map(mondrian_q).fillna(q_global)

    out_cols = [
        "week_start", "sku_id", "region_id", "forecasted_demand",
        "lower_bound", "upper_bound", "interval_width",
        "coverage_target", "q_hat", "velocity_class",
    ]
    forecast[out_cols].to_csv(processed / "forecast_intervals.csv", index=False)

    # ── Calibration metrics CSV ──────────────────────────────────────────────
    cov_rows = []
    for vc, grp in backtest.groupby("velocity_class"):
        q = mondrian_q[str(vc)]
        emp = float(grp["covered_mondrian"].mean()) if "covered_mondrian" in grp else float("nan")
        cov_rows.append({
            "velocity_class": vc,
            "n_calibration_points": len(grp),
            "q_hat": round(q, 4),
            "empirical_coverage": round(emp, 4),
            "target_coverage": round(1 - _ALPHA, 4),
            "coverage_gap": round(emp - (1 - _ALPHA), 4),
        })
    pd.DataFrame(cov_rows).to_csv(
        processed / "conformal_calibration_metrics.csv", index=False
    )

    # Adaptive coverage alert: if empirical coverage < 85% something is wrong
    coverage_ok = empirical_mondrian >= 0.85
    print(
        f"  Conformal prediction: global q̂={q_global:.2f}, "
        f"empirical coverage={empirical_global:.1%} (target={1-_ALPHA:.0%}). "
        f"Mondrian coverage={empirical_mondrian:.1%}. "
        f"{'OK' if coverage_ok else 'WARNING: coverage below threshold'}"
    )
    print(
        f"  Mondrian q̂ → "
        + ", ".join(f"{k}={v:.2f}" for k, v in sorted(mondrian_q.items()))
    )

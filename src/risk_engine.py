"""
Monte Carlo Risk Engine: Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR).

Most supply chain tools report expected costs. Risk management cares about the
TAIL — how bad does it get in the worst 5% of scenarios?

VaR_α:   Minimum cost threshold exceeded only (1−α)% of the time.
CVaR_α:  Expected cost GIVEN that cost exceeds VaR_α.  (aka Expected Shortfall)

CVaR is coherent (Artzner et al. 1999) and is the industry standard for
quantifying inventory risk in insurance, banking, and operations.

Algorithm per (SKU, warehouse) pair
-------------------------------------
1. Fit demand distribution to historical weekly actuals.
   Use Negative Binomial (over-dispersed counts) when variance > mean,
   else Poisson.  Fall back to Normal for fast SKUs.

2. Sample N_SIM = 10,000 demand trajectories over the forecast horizon.

3. For each trajectory: simulate a (reorder-point, EOQ) policy and compute
   total horizon cost (ordering + holding + stockout).

4. Compute:
   - E[cost], Std[cost]
   - VaR_95   = 95th percentile of simulated cost distribution
   - CVaR_95  = mean of top 5% costs  (Expected Shortfall)
   - Worst-case cost
   - Probability of stockout ≥ 1 unit across horizon

Why CVaR beats VaR
-------------------
VaR says "losses won't exceed X with 95% probability."
CVaR says "given that losses exceed X, they average Y."
CVaR captures tail shape; VaR ignores severity above the threshold.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_N_SIM = 10_000
_ALPHA = 0.95      # CVaR confidence level
_STOCKOUT_MULT = 3


def _fit_distribution(weekly_demands: np.ndarray):
    """Fit best distribution. Returns (dist_name, sample_fn)."""
    mu = float(np.mean(weekly_demands))
    var = float(np.var(weekly_demands))

    if mu <= 0:
        return "degenerate", lambda n: np.zeros(n)

    if var > mu * 1.5:
        # Negative Binomial: p = mu/var, r = mu²/(var-mu)
        p = min(max(mu / var, 1e-6), 1.0 - 1e-6)
        r = max(mu * p / (1 - p), 0.5)
        def _nb(n, _r=r, _p=p):
            return np.random.negative_binomial(_r, _p, n).astype(float)
        return "NegBinomial", _nb
    else:
        lam = max(mu, 0.1)
        def _pois(n, _l=lam):
            return np.random.poisson(_l, n).astype(float)
        return "Poisson", _pois


def _simulate_policy_cost(
    demand_matrix: np.ndarray,   # shape (N_SIM, T)
    I0: float,
    reorder_point: float,
    eoq: float,
    K: float,
    h: float,
    p: float,
) -> np.ndarray:
    """Simulate (s, Q) policy across N_SIM scenarios. Returns cost per scenario."""
    N, T = demand_matrix.shape
    costs = np.zeros(N)
    inv = np.full(N, I0)
    in_transit = np.zeros(N)     # units on order (simplified: 1-period lead time)

    for t in range(T):
        # Receive in-transit orders
        inv += in_transit

        # Fulfil demand
        fulfilled = np.minimum(inv, demand_matrix[:, t])
        shortfall = demand_matrix[:, t] - fulfilled
        inv -= fulfilled

        # Stockout cost
        costs += p * shortfall

        # Holding cost on remaining inventory
        costs += h * np.maximum(inv, 0)

        # Place order if below reorder point
        order_mask = inv <= reorder_point
        costs += K * order_mask.astype(float)
        in_transit = eoq * order_mask.astype(float)

    return costs


def run_risk_engine(root: Path) -> None:
    processed = root / "data" / "processed"
    raw = root / "data" / "raw"

    forecast = pd.read_csv(processed / "forecast_output.csv", parse_dates=["week_start"])
    orders = pd.read_csv(raw / "orders.csv", parse_dates=["week_start"])
    products = pd.read_csv(raw / "products.csv")
    inventory = pd.read_csv(raw / "inventory_snapshot.csv")
    lead_times = pd.read_csv(raw / "supplier_lead_times.csv")
    inventory_recs = pd.read_csv(processed / "inventory_recommendations.csv")

    # Aggregate forecast to (sku, warehouse) using historical shares
    hist = orders.groupby(["sku_id", "region_id", "warehouse_id"], as_index=False)["order_units"].sum()
    totals = hist.groupby(["sku_id", "region_id"])["order_units"].transform("sum")
    hist["share"] = hist["order_units"] / totals.replace(0, 1)

    fc_wh = forecast.merge(hist[["sku_id", "region_id", "warehouse_id", "share"]], on=["sku_id", "region_id"], how="left")
    fc_wh["share"] = fc_wh["share"].fillna(0)
    fc_wh["wh_demand"] = fc_wh["forecasted_demand"] * fc_wh["share"]

    fc_agg = (
        fc_wh.groupby(["sku_id", "warehouse_id", "week_start"], as_index=False)["wh_demand"].sum()
    )

    # Historical weekly demand for fitting distributions
    hist_weekly = (
        orders.groupby(["sku_id", "warehouse_id", "week_start"], as_index=False)["order_units"].sum()
    )

    inv_map = inventory.set_index(["sku_id", "warehouse_id"])
    lt_map = lead_times.set_index(["sku_id", "warehouse_id"])
    prod_map = products.set_index("sku_id")
    rec_map = inventory_recs.set_index(["sku_id", "warehouse_id"])

    rng_seed = 42
    np.random.seed(rng_seed)

    risk_rows: list[dict] = []

    pairs = fc_agg[["sku_id", "warehouse_id"]].drop_duplicates()

    for _, pair in pairs.iterrows():
        sku, wh = pair["sku_id"], pair["warehouse_id"]

        if sku not in prod_map.index:
            continue

        prod = prod_map.loc[sku]
        K = float(prod["order_cost"])
        unit_cost = float(prod["unit_cost"])
        h = unit_cost * float(prod["holding_cost_pct"]) / 52.0
        p_cost = unit_cost * _STOCKOUT_MULT

        # Historical demand for this pair
        hist_d = hist_weekly[
            (hist_weekly["sku_id"] == sku) & (hist_weekly["warehouse_id"] == wh)
        ]["order_units"].values.astype(float)

        if len(hist_d) < 4:
            continue

        dist_name, sample_fn = _fit_distribution(hist_d)

        # Horizon demand matrix (N_SIM × T)
        T = int((fc_agg[(fc_agg["sku_id"] == sku) & (fc_agg["warehouse_id"] == wh)]["week_start"].nunique()))
        T = max(T, 1)
        demand_matrix = np.column_stack([sample_fn(_N_SIM) for _ in range(T)])

        # Current inventory and policy parameters
        inv_key = (sku, wh)
        I0 = 0.0
        if inv_key in inv_map.index:
            r = inv_map.loc[inv_key]
            I0 = max(0.0, float(r["current_inventory_units"]) - float(r["reserved_inventory_units"]))

        reorder_pt = 0.0
        eoq = float(np.sqrt(2 * max(hist_d.mean() * 52, 1) * K / max(h, 1e-6)))
        if inv_key in rec_map.index:
            rec = rec_map.loc[inv_key]
            reorder_pt = float(rec.get("reorder_point_units", reorder_pt))
            if "eoq_units" in rec.index:
                eoq = float(rec["eoq_units"]) if float(rec["eoq_units"]) > 0 else eoq

        # Monte Carlo simulation
        scenario_costs = _simulate_policy_cost(demand_matrix, I0, reorder_pt, eoq, K, h, p_cost)

        # Risk metrics
        mean_cost = float(np.mean(scenario_costs))
        std_cost = float(np.std(scenario_costs))
        var_95 = float(np.percentile(scenario_costs, _ALPHA * 100))
        cvar_95 = float(np.mean(scenario_costs[scenario_costs >= var_95]))
        worst_case = float(np.max(scenario_costs))
        best_case = float(np.min(scenario_costs))

        # Probability of experiencing at least one stockout
        stockout_prob = float(np.mean(np.any(demand_matrix > (np.full((_N_SIM, T), I0)), axis=1)))

        risk_rows.append({
            "sku_id": sku,
            "warehouse_id": wh,
            "demand_distribution": dist_name,
            "n_simulations": _N_SIM,
            "horizon_weeks": T,
            "mean_horizon_cost": round(mean_cost, 2),
            "std_horizon_cost": round(std_cost, 2),
            "var_95": round(var_95, 2),
            "cvar_95": round(cvar_95, 2),
            "worst_case_cost": round(worst_case, 2),
            "best_case_cost": round(best_case, 2),
            "cvar_to_mean_ratio": round(cvar_95 / max(mean_cost, 1), 3),
            "stockout_probability": round(stockout_prob, 4),
            "risk_tier": (
                "Critical" if stockout_prob > 0.40 else
                "High" if stockout_prob > 0.20 else
                "Medium" if stockout_prob > 0.10 else
                "Low"
            ),
        })

    risk_df = pd.DataFrame(risk_rows)
    risk_df.to_csv(processed / "risk_metrics.csv", index=False)

    n_critical = int((risk_df["risk_tier"] == "Critical").sum())
    avg_cvar = risk_df["cvar_95"].mean()
    avg_cvar_ratio = risk_df["cvar_to_mean_ratio"].mean()

    print(
        f"  Monte Carlo risk engine: {len(risk_df)} SKU-warehouse pairs, "
        f"{_N_SIM:,} simulations each, "
        f"{n_critical} critical-tier, "
        f"avg CVaR_95=${avg_cvar:,.0f} ({avg_cvar_ratio:.1f}× expected cost)."
    )

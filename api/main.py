"""
Supply Chain Intelligence REST API — FastAPI

Run with:
    uvicorn api.main:app --reload --port 8000

All data is loaded once at startup from processed CSVs; no DB queries per
request.  This mirrors the read-heavy, low-latency pattern of a real planning
API sitting behind a BI dashboard.

Endpoints
---------
GET  /health                           → service health + data freshness
GET  /forecast/{sku_id}                → point forecast + conformal intervals
GET  /forecast/{sku_id}/intervals      → full interval table for a SKU
GET  /inventory/recommendations        → EOQ-based recommendations (filtered)
GET  /inventory/mip/{sku_id}           → MIP order schedule for a SKU
GET  /resilience/scores                → per-node resilience scores
GET  /resilience/spof                  → single points of failure only
GET  /risk/summary                     → CVaR risk metrics (top-N by tier)
GET  /risk/{sku_id}/{warehouse_id}     → full risk profile for one pair
GET  /network/edges                    → supply network edge list
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"

app = FastAPI(
    title="Supply Chain Intelligence API",
    description=(
        "Demand forecasting with conformal prediction intervals, "
        "MIP inventory optimization, network resilience scoring, "
        "and Monte Carlo CVaR risk analysis."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ── Data loaded once at startup ──────────────────────────────────────────────
_data: dict[str, pd.DataFrame] = {}


@app.on_event("startup")
def _load_data() -> None:
    files = {
        "intervals": "forecast_intervals.csv",
        "forecast": "forecast_output.csv",
        "inventory": "inventory_recommendations.csv",
        "mip_schedule": "mip_order_schedule.csv",
        "mip_summary": "mip_inventory_summary.csv",
        "resilience": "resilience_scores.csv",
        "edges": "network_edges.csv",
        "risk": "risk_metrics.csv",
        "calibration": "conformal_calibration_metrics.csv",
    }
    for key, fname in files.items():
        path = PROCESSED / fname
        if path.exists():
            _data[key] = pd.read_csv(path)
        else:
            _data[key] = pd.DataFrame()


# ── Response models ───────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    datasets_loaded: int
    datasets_missing: list[str]


class ForecastPoint(BaseModel):
    week_start: str
    region_id: str
    forecasted_demand: float
    lower_bound: float
    upper_bound: float
    interval_width: float
    coverage_target: float


class RiskProfile(BaseModel):
    sku_id: str
    warehouse_id: str
    demand_distribution: str
    mean_horizon_cost: float
    var_95: float
    cvar_95: float
    cvar_to_mean_ratio: float
    stockout_probability: float
    risk_tier: str


class ResilienceScore(BaseModel):
    node_id: str
    node_type: str
    resilience_score: float
    flow_loss_pct: float
    betweenness_centrality: float
    is_single_point_of_failure: bool


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    missing = [k for k, df in _data.items() if df.empty]
    return HealthResponse(
        status="healthy" if not missing else "degraded",
        datasets_loaded=len(_data) - len(missing),
        datasets_missing=missing,
    )


@app.get("/forecast/{sku_id}", response_model=list[ForecastPoint], tags=["Forecast"])
def get_forecast(
    sku_id: str,
    region_id: Optional[str] = Query(None, description="Filter by region"),
):
    """Demand forecast with conformal prediction intervals for a SKU."""
    df = _data.get("intervals", pd.DataFrame())
    if df.empty:
        raise HTTPException(503, "Forecast intervals not available. Run pipeline first.")

    mask = df["sku_id"] == sku_id
    if region_id:
        mask &= df["region_id"] == region_id

    rows = df[mask]
    if rows.empty:
        raise HTTPException(404, f"SKU '{sku_id}' not found.")

    return [
        ForecastPoint(
            week_start=str(r["week_start"]),
            region_id=str(r["region_id"]),
            forecasted_demand=float(r["forecasted_demand"]),
            lower_bound=float(r["lower_bound"]),
            upper_bound=float(r["upper_bound"]),
            interval_width=float(r["interval_width"]),
            coverage_target=float(r["coverage_target"]),
        )
        for _, r in rows.iterrows()
    ]


@app.get("/forecast/calibration/metrics", tags=["Forecast"])
def get_calibration_metrics():
    """Conformal calibration metrics per velocity class."""
    df = _data.get("calibration", pd.DataFrame())
    if df.empty:
        raise HTTPException(503, "Calibration metrics not available.")
    return df.to_dict(orient="records")


@app.get("/inventory/recommendations", tags=["Inventory"])
def get_inventory_recommendations(
    stockout_risk: Optional[str] = Query(None, description="Filter: High or Low"),
    category: Optional[str] = Query(None),
    limit: int = Query(50, le=500),
):
    """EOQ-based inventory recommendations with risk flags."""
    df = _data.get("inventory", pd.DataFrame())
    if df.empty:
        raise HTTPException(503, "Inventory data not available.")
    if stockout_risk:
        df = df[df["stockout_risk"] == stockout_risk]
    if category:
        df = df[df["category"] == category]
    return df.head(limit).to_dict(orient="records")


@app.get("/inventory/mip/{sku_id}", tags=["Inventory"])
def get_mip_schedule(sku_id: str, warehouse_id: Optional[str] = Query(None)):
    """MIP-optimal multi-period ordering schedule for a SKU."""
    sched = _data.get("mip_schedule", pd.DataFrame())
    summ = _data.get("mip_summary", pd.DataFrame())

    if sched.empty:
        raise HTTPException(503, "MIP schedule not available. Run pipeline first.")

    mask = sched["sku_id"] == sku_id
    smask = summ["sku_id"] == sku_id
    if warehouse_id:
        mask &= sched["warehouse_id"] == warehouse_id
        smask &= summ["warehouse_id"] == warehouse_id

    if sched[mask].empty:
        raise HTTPException(404, f"SKU '{sku_id}' not found in MIP schedule.")

    return {
        "schedule": sched[mask].to_dict(orient="records"),
        "summary": summ[smask].to_dict(orient="records"),
    }


@app.get("/resilience/scores", response_model=list[ResilienceScore], tags=["Resilience"])
def get_resilience_scores(node_type: Optional[str] = Query(None)):
    """Per-node resilience scores and single-point-of-failure flags."""
    df = _data.get("resilience", pd.DataFrame())
    if df.empty:
        raise HTTPException(503, "Resilience data not available.")
    if node_type:
        df = df[df["node_type"] == node_type]
    return [
        ResilienceScore(
            node_id=str(r["node_id"]),
            node_type=str(r["node_type"]),
            resilience_score=float(r["resilience_score"]),
            flow_loss_pct=float(r["flow_loss_pct"]),
            betweenness_centrality=float(r["betweenness_centrality"]),
            is_single_point_of_failure=bool(r["is_single_point_of_failure"]),
        )
        for _, r in df.iterrows()
    ]


@app.get("/resilience/spof", tags=["Resilience"])
def get_single_points_of_failure():
    """Nodes whose removal causes >50% throughput loss."""
    df = _data.get("resilience", pd.DataFrame())
    if df.empty:
        raise HTTPException(503, "Resilience data not available.")
    spof = df[df["is_single_point_of_failure"] == True]
    return spof.to_dict(orient="records")


@app.get("/network/edges", tags=["Network"])
def get_network_edges(edge_type: Optional[str] = Query(None)):
    """Supply chain network edges with capacities."""
    df = _data.get("edges", pd.DataFrame())
    if df.empty:
        raise HTTPException(503, "Network edge data not available.")
    if edge_type:
        df = df[df["edge_type"] == edge_type]
    return df.to_dict(orient="records")


@app.get("/risk/summary", tags=["Risk"])
def get_risk_summary(
    risk_tier: Optional[str] = Query(None, description="Critical / High / Medium / Low"),
    limit: int = Query(50, le=500),
):
    """Monte Carlo CVaR risk summary, sorted by CVaR descending."""
    df = _data.get("risk", pd.DataFrame())
    if df.empty:
        raise HTTPException(503, "Risk data not available.")
    if risk_tier:
        df = df[df["risk_tier"] == risk_tier]
    df = df.sort_values("cvar_95", ascending=False)
    return df.head(limit).to_dict(orient="records")


@app.get("/risk/{sku_id}/{warehouse_id}", response_model=RiskProfile, tags=["Risk"])
def get_risk_profile(sku_id: str, warehouse_id: str):
    """Full Monte Carlo risk profile for a specific SKU-warehouse pair."""
    df = _data.get("risk", pd.DataFrame())
    if df.empty:
        raise HTTPException(503, "Risk data not available.")
    rows = df[(df["sku_id"] == sku_id) & (df["warehouse_id"] == warehouse_id)]
    if rows.empty:
        raise HTTPException(404, f"No risk data for {sku_id} / {warehouse_id}.")
    r = rows.iloc[0]
    return RiskProfile(
        sku_id=str(r["sku_id"]),
        warehouse_id=str(r["warehouse_id"]),
        demand_distribution=str(r["demand_distribution"]),
        mean_horizon_cost=float(r["mean_horizon_cost"]),
        var_95=float(r["var_95"]),
        cvar_95=float(r["cvar_95"]),
        cvar_to_mean_ratio=float(r["cvar_to_mean_ratio"]),
        stockout_probability=float(r["stockout_probability"]),
        risk_tier=str(r["risk_tier"]),
    )

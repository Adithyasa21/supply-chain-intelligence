from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
PBI = ROOT / "data" / "powerbi_outputs"

st.set_page_config(
    page_title="Supply Chain Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def _azure_status() -> dict:
    """Check Azure connectivity once at startup."""
    try:
        from cloud.config import azure_config
        from cloud.storage import DataLakeStore
        store = DataLakeStore.from_config(azure_config) if azure_config.enabled else None
        return {
            "configured": azure_config.enabled,
            "storage_account": azure_config.storage_account_name or "—",
            "storage_connected": store is not None,
            "ml_configured": azure_config.ml_configured,
            "ml_workspace": azure_config.ml_workspace_name or "—",
            "keyvault_configured": azure_config.keyvault_configured,
        }
    except Exception:
        return {"configured": False, "storage_connected": False,
                "ml_configured": False, "keyvault_configured": False,
                "storage_account": "—", "ml_workspace": "—"}

st.markdown("""
<style>
.metric-card { background:#1e2130; border-radius:8px; padding:16px; }
.risk-critical { color:#ff4b4b; font-weight:bold; }
.risk-high { color:#ffa500; font-weight:bold; }
.risk-low { color:#00cc88; font-weight:bold; }
</style>
""", unsafe_allow_html=True)

st.title("Supply Chain Intelligence Platform")
st.caption(
    "Demand forecasting · Conformal prediction intervals · MIP inventory optimization · "
    "Network resilience · Monte Carlo CVaR risk · REST API"
)


# ── Data loaders ─────────────────────────────────────────────────────────────
@st.cache_data
def load(name: str, folder: Path = PROCESSED) -> pd.DataFrame:
    p = folder / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def require(df: pd.DataFrame, msg: str) -> bool:
    if df.empty:
        st.warning(msg)
        return False
    return True


# Core data
intervals = load("forecast_intervals")
mip_schedule = load("mip_order_schedule")
mip_summary = load("mip_inventory_summary")
resilience = load("resilience_scores")
edges = load("network_edges")
risk = load("risk_metrics")
calibration = load("conformal_calibration_metrics")
inventory = load("inventory_recommendations")
scenario = load("scenario_results", PBI)
executive_raw = load("executive_summary", PBI)
monthly = load("monthly_kpis", PBI)
feature_imp = load("forecast_feature_importance_top30", PBI)
backtest = load("forecast_backtest")
products = load("products", RAW)
warehouses = load("warehouses", RAW)

# ── Sidebar: Azure status panel ──────────────────────────────────────────────
az = _azure_status()
st.sidebar.header("Azure Status")
if az["configured"] and az["storage_connected"]:
    st.sidebar.success(f"ADLS Gen2: {az['storage_account']}")
elif az["configured"]:
    st.sidebar.warning("ADLS Gen2: configured but unreachable")
else:
    st.sidebar.info("Azure: local mode (set AZURE_STORAGE_ACCOUNT_NAME to connect)")

if az["ml_configured"]:
    st.sidebar.success(f"Azure ML: {az['ml_workspace']}")
else:
    st.sidebar.info("Azure ML: not configured")

if az["keyvault_configured"]:
    st.sidebar.success("Key Vault: connected")

st.sidebar.caption("Run `infra/provision.sh` to set up Azure resources.")

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.header("Filters")

sku_list = sorted(intervals["sku_id"].unique().tolist()) if not intervals.empty else []
selected_sku = st.sidebar.selectbox("SKU", sku_list) if sku_list else None

region_list = ["All"] + (sorted(intervals["region_id"].unique().tolist()) if not intervals.empty else [])
selected_region = st.sidebar.selectbox("Region", region_list)

category_list = ["All"] + (sorted(inventory["category"].dropna().unique().tolist()) if not inventory.empty else [])
selected_category = st.sidebar.selectbox("Category", category_list)

wh_list = ["All"] + (sorted(mip_summary["warehouse_id"].unique().tolist()) if not mip_summary.empty else [])
selected_wh = st.sidebar.selectbox("Warehouse", wh_list)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "Executive Overview",
    "Forecast + Conformal Intervals",
    "MIP Inventory Optimization",
    "Network Resilience",
    "Monte Carlo Risk (CVaR)",
    "Azure Platform",
    "API Reference",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Executive Overview
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    if not executive_raw.empty:
        ex = executive_raw.iloc[0]
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Total Orders", f"{int(ex['total_orders']):,}")
        c2.metric("Revenue", f"${ex['total_revenue']:,.0f}")
        c3.metric("Forecast MAPE", f"{ex['forecast_mape_pct']:.1f}%")
        c4.metric("Stockout-Risk Pairs", f"{int(ex['stockout_risk_pairs']):,}")
        c5.metric("Replenishment Cost", f"${ex['estimated_replenishment_cost']:,.0f}")
        if not risk.empty:
            n_crit = int((risk["risk_tier"] == "Critical").sum())
            c6.metric("Critical Risk SKUs", f"{n_crit}", delta="CVaR-flagged", delta_color="inverse")

    col_l, col_r = st.columns(2)
    with col_l:
        if not monthly.empty:
            st.plotly_chart(
                px.line(monthly, x="month", y=["units", "orders"], markers=True,
                        title="Monthly Order Volume & Units"),
                use_container_width=True,
            )
    with col_r:
        if not monthly.empty:
            st.plotly_chart(
                px.line(monthly, x="month", y="late_delivery_rate_pct", markers=True,
                        title="Late Delivery Rate %",
                        color_discrete_sequence=["#ff6b6b"]),
                use_container_width=True,
            )

    if not feature_imp.empty:
        st.plotly_chart(
            px.bar(feature_imp.head(15), x="importance", y="feature", orientation="h",
                   title="Top 15 Demand Forecast Feature Importances",
                   color="importance", color_continuous_scale="Blues"),
            use_container_width=True,
        )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Forecast + Conformal Prediction Intervals
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("### Demand Forecast with Calibrated Prediction Intervals")
    st.info(
        "Intervals use **Mondrian Split Conformal Prediction** (Vovk et al. 2005). "
        "Unlike ±2σ bands, these have a **provable finite-sample coverage guarantee**: "
        "P(actual ∈ [lower, upper]) ≥ 90%, regardless of model quality or demand distribution."
    )

    if not calibration.empty:
        st.markdown("#### Calibration metrics by velocity class")
        cal_display = calibration.copy()
        cal_display["coverage_gap"] = cal_display["empirical_coverage"] - cal_display["target_coverage"]
        cal_display["Status"] = cal_display["coverage_gap"].apply(
            lambda x: "✓ Valid" if x >= -0.02 else "⚠ Undercoverage"
        )
        st.dataframe(
            cal_display.style.format({
                "q_hat": "{:.2f}",
                "empirical_coverage": "{:.1%}",
                "target_coverage": "{:.1%}",
                "coverage_gap": "{:+.1%}",
            }),
            use_container_width=True,
        )

    if not intervals.empty and selected_sku:
        sku_data = intervals[intervals["sku_id"] == selected_sku].copy()
        if selected_region != "All":
            sku_data = sku_data[sku_data["region_id"] == selected_region]

        if not sku_data.empty:
            sku_data = sku_data.sort_values("week_start")
            regions_in_data = sku_data["region_id"].unique().tolist()

            st.markdown(f"#### {selected_sku} — Forecast with 90% Prediction Intervals")

            fig = go.Figure()
            colors = px.colors.qualitative.Set2

            for i, reg in enumerate(regions_in_data):
                sub = sku_data[sku_data["region_id"] == reg].sort_values("week_start")
                color = colors[i % len(colors)]

                # Shaded interval band
                fig.add_trace(go.Scatter(
                    x=list(sub["week_start"]) + list(sub["week_start"])[::-1],
                    y=list(sub["upper_bound"]) + list(sub["lower_bound"])[::-1],
                    fill="toself",
                    fillcolor=color.replace("rgb", "rgba").replace(")", ", 0.15)"),
                    line=dict(color="rgba(255,255,255,0)"),
                    name=f"{reg} 90% interval",
                    showlegend=True,
                ))
                # Point forecast line
                fig.add_trace(go.Scatter(
                    x=sub["week_start"],
                    y=sub["forecasted_demand"],
                    mode="lines+markers",
                    name=f"{reg} forecast",
                    line=dict(color=color, width=2),
                ))

            fig.update_layout(
                title=f"{selected_sku}: Demand Forecast + Conformal Intervals",
                xaxis_title="Week",
                yaxis_title="Units",
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

            avg_width = sku_data["interval_width"].mean()
            st.caption(
                f"Average interval width: **{avg_width:.1f} units** across all regions. "
                f"Narrower intervals = more predictable SKU."
            )

    if not backtest.empty and not intervals.empty:
        st.markdown("#### Backtest: actual vs forecast sample (most recent 100 rows)")
        bt = backtest.sort_values("week_start", ascending=False).head(100)
        st.dataframe(
            bt.style.format({
                "actual_demand": "{:.1f}",
                "forecasted_demand": "{:.1f}",
                "absolute_error": "{:.1f}",
                "ape": "{:.2%}",
            }),
            use_container_width=True,
        )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MIP Inventory Optimization
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("### Multi-Period Inventory Lot-Sizing (MIP)")
    st.info(
        "Replaces the static EOQ model with a **Mixed-Integer Program** (Capacitated Lot-Sizing, CLSP). "
        "The MIP plans an optimal ordering schedule across the full 8-week forecast horizon, "
        "explicitly modeling **fixed ordering costs**, **lead-time delays**, and "
        "**multi-period demand profiles** — EOQ cannot do this."
    )

    if not mip_summary.empty:
        summary = mip_summary.copy()
        if selected_wh != "All":
            summary = summary[summary["warehouse_id"] == selected_wh]

        total_mip = summary["mip_total_cost"].sum()
        total_eoq = summary["eoq_benchmark_cost"].sum()
        total_savings = total_mip - total_eoq  # negative = MIP cheaper
        n_optimal = int((summary["mip_status"] == "Optimal").sum())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MIP Total Cost", f"${total_mip:,.0f}")
        c2.metric("EOQ Benchmark", f"${total_eoq:,.0f}")
        c3.metric(
            "Savings vs EOQ",
            f"${abs(total_savings):,.0f}",
            delta=f"{'MIP cheaper' if total_savings < 0 else 'EOQ cheaper'}",
            delta_color="normal" if total_savings < 0 else "inverse",
        )
        c4.metric("Optimal Solutions", f"{n_optimal}/{len(summary)}")

        col_l, col_r = st.columns(2)
        with col_l:
            top_savings = summary.nlargest(15, "cost_savings_vs_eoq")
            st.plotly_chart(
                px.bar(top_savings, x="sku_id", y="cost_savings_vs_eoq",
                       color="warehouse_id",
                       title="Top SKUs by MIP Cost Savings vs EOQ ($)",
                       labels={"cost_savings_vs_eoq": "Savings ($)"}),
                use_container_width=True,
            )
        with col_r:
            st.plotly_chart(
                px.scatter(
                    summary,
                    x="eoq_benchmark_cost",
                    y="mip_total_cost",
                    color="warehouse_id",
                    hover_data=["sku_id", "n_orders_planned", "lead_time_weeks"],
                    title="MIP Cost vs EOQ Benchmark (points below diagonal = MIP wins)",
                    labels={"eoq_benchmark_cost": "EOQ Cost ($)", "mip_total_cost": "MIP Cost ($)"},
                ),
                use_container_width=True,
            )
            x_range = [0, float(summary[["eoq_benchmark_cost", "mip_total_cost"]].max().max())]

        if not mip_schedule.empty and selected_sku:
            st.markdown(f"#### {selected_sku} — MIP Order Schedule")
            sku_sched = mip_schedule[mip_schedule["sku_id"] == selected_sku].copy()
            if selected_wh != "All":
                sku_sched = sku_sched[sku_sched["warehouse_id"] == selected_wh]

            if not sku_sched.empty:
                for wh_id, wh_data in sku_sched.groupby("warehouse_id"):
                    wh_data = wh_data.sort_values("period")
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=wh_data["week_start"],
                        y=wh_data["order_quantity"],
                        name="Order Quantity",
                        marker_color="#4C72B0",
                    ))
                    fig.add_trace(go.Scatter(
                        x=wh_data["week_start"],
                        y=wh_data["projected_inventory"],
                        name="Projected Inventory",
                        yaxis="y2",
                        line=dict(color="#DD8452", width=2),
                    ))
                    fig.add_trace(go.Scatter(
                        x=wh_data["week_start"],
                        y=wh_data["forecasted_demand"],
                        name="Demand",
                        yaxis="y2",
                        line=dict(color="#55A868", width=2, dash="dot"),
                    ))
                    fig.update_layout(
                        title=f"{selected_sku} @ {wh_id} — Order Schedule + Inventory Projection",
                        yaxis=dict(title="Order Quantity (units)"),
                        yaxis2=dict(title="Inventory / Demand", overlaying="y", side="right"),
                        barmode="group",
                        legend=dict(orientation="h"),
                    )
                    st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Full MIP Summary Table")
        display_cols = [
            "sku_id", "warehouse_id", "mip_total_cost", "eoq_benchmark_cost",
            "cost_savings_vs_eoq", "savings_pct", "n_orders_planned",
            "lead_time_weeks", "mip_status",
        ]
        st.dataframe(
            summary[display_cols].sort_values("cost_savings_vs_eoq", ascending=False)
            .style.format({
                "mip_total_cost": "${:,.0f}",
                "eoq_benchmark_cost": "${:,.0f}",
                "cost_savings_vs_eoq": "${:,.0f}",
                "savings_pct": "{:.1f}%",
            }),
            use_container_width=True,
        )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Network Resilience
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("### Supply Chain Network Resilience")
    st.info(
        "Models the supply network as a **directed flow graph** (Suppliers → Warehouses → Regions). "
        "For each node: simulates its removal and computes **R(v) = residual flow / baseline flow**. "
        "R(v) < 50% → Single Point of Failure (SPOF). "
        "Uses NetworkX max-flow (push-relabel algorithm)."
    )

    if not resilience.empty:
        spof = resilience[resilience["is_single_point_of_failure"] == True]
        n_spof = len(spof)
        min_r = resilience["resilience_score"].min()
        avg_r = resilience["resilience_score"].mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Nodes", len(resilience))
        c2.metric("Single Points of Failure", n_spof, delta="critical" if n_spof > 0 else "none", delta_color="inverse" if n_spof > 0 else "off")
        c3.metric("Min Resilience Score", f"{min_r:.1%}")
        c4.metric("Avg Resilience Score", f"{avg_r:.1%}")

        col_l, col_r = st.columns(2)
        with col_l:
            fig_r = px.bar(
                resilience.sort_values("resilience_score"),
                x="resilience_score",
                y="node_id",
                color="node_type",
                orientation="h",
                title="Node Resilience Scores (lower = more critical)",
                labels={"resilience_score": "R(v) — Fraction of Flow Retained"},
                color_discrete_map={"supplier": "#4C72B0", "warehouse": "#DD8452", "region": "#55A868"},
            )
            fig_r.add_vline(x=0.5, line_dash="dash", line_color="red",
                            annotation_text="SPOF threshold (50%)")
            st.plotly_chart(fig_r, use_container_width=True)

        with col_r:
            fig_bc = px.scatter(
                resilience,
                x="betweenness_centrality",
                y="flow_loss_pct",
                color="node_type",
                size="flow_loss_pct",
                hover_data=["node_id", "resilience_score"],
                title="Betweenness Centrality vs Flow Loss % on Failure",
                labels={
                    "betweenness_centrality": "Betweenness Centrality (routing importance)",
                    "flow_loss_pct": "Flow Loss % if Node Removed",
                },
            )
            st.plotly_chart(fig_bc, use_container_width=True)

        # Network graph visualization
        if not edges.empty:
            st.markdown("#### Supply Network Graph")
            node_positions: dict[str, tuple[float, float]] = {}
            node_colors: dict[str, str] = {}

            # Layout: suppliers left, warehouses center, regions right
            res_lookup = resilience.set_index("node_id")

            supplier_nodes = resilience[resilience["node_type"] == "supplier"]["node_id"].tolist()
            wh_nodes = resilience[resilience["node_type"] == "warehouse"]["node_id"].tolist()
            region_nodes = resilience[resilience["node_type"] == "region"]["node_id"].tolist()

            for i, n in enumerate(supplier_nodes):
                node_positions[n] = (0.0, (i + 0.5) / max(len(supplier_nodes), 1))
            for i, n in enumerate(wh_nodes):
                node_positions[n] = (0.5, (i + 0.5) / max(len(wh_nodes), 1))
            for i, n in enumerate(region_nodes):
                node_positions[n] = (1.0, (i + 0.5) / max(len(region_nodes), 1))

            edge_x, edge_y = [], []
            for _, e in edges.iterrows():
                s, t = str(e["source"]), str(e["target"])
                if s in node_positions and t in node_positions:
                    x0, y0 = node_positions[s]
                    x1, y1 = node_positions[t]
                    edge_x += [x0, x1, None]
                    edge_y += [y0, y1, None]

            edge_trace = go.Scatter(
                x=edge_x, y=edge_y, mode="lines",
                line=dict(width=0.8, color="#888"),
                hoverinfo="none",
            )

            all_nodes = supplier_nodes + wh_nodes + region_nodes
            node_x = [node_positions[n][0] for n in all_nodes if n in node_positions]
            node_y = [node_positions[n][1] for n in all_nodes if n in node_positions]
            node_color = []
            node_size = []
            node_text = []
            for n in all_nodes:
                if n not in node_positions:
                    continue
                r_score = float(res_lookup.loc[n, "resilience_score"]) if n in res_lookup.index else 1.0
                is_spof_node = bool(res_lookup.loc[n, "is_single_point_of_failure"]) if n in res_lookup.index else False
                node_color.append("#ff4b4b" if is_spof_node else "#00cc88")
                node_size.append(28 if is_spof_node else 18)
                node_text.append(f"{n}<br>R={r_score:.0%}")

            node_trace = go.Scatter(
                x=node_x, y=node_y, mode="markers+text",
                marker=dict(size=node_size, color=node_color, line=dict(width=1, color="#333")),
                text=[n for n in all_nodes if n in node_positions],
                textposition="top center",
                hovertext=node_text,
                hoverinfo="text",
            )

            fig_net = go.Figure(
                data=[edge_trace, node_trace],
                layout=go.Layout(
                    title="Supply Network Topology (red = SPOF, green = resilient)",
                    showlegend=False,
                    hovermode="closest",
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                               tickvals=[0, 0.5, 1], ticktext=["Suppliers", "Warehouses", "Regions"]),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=500,
                )
            )
            fig_net.update_xaxes(tickvals=[0, 0.5, 1], ticktext=["Suppliers", "Warehouses", "Regions"], showticklabels=True)
            st.plotly_chart(fig_net, use_container_width=True)

        if n_spof > 0:
            st.error(f"{n_spof} single point(s) of failure detected:")
            st.dataframe(spof[["node_id", "node_type", "resilience_score", "flow_loss_pct", "betweenness_centrality"]], use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Monte Carlo Risk (CVaR)
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("### Monte Carlo Inventory Risk — VaR and CVaR")
    st.info(
        "**10,000 demand scenarios** sampled per (SKU, warehouse) from fitted distributions "
        "(Negative Binomial or Poisson). For each scenario: simulate a (s, Q) policy and compute "
        "total 8-week inventory cost.\n\n"
        "**VaR₉₅**: Cost exceeded in only 5% of scenarios.\n\n"
        "**CVaR₉₅** (Expected Shortfall): Average cost in the worst 5% — the actuarial gold standard "
        "for tail risk, coherent in the sense of Artzner et al. (1999)."
    )

    if not risk.empty:
        risk_display = risk.copy()
        if selected_wh != "All":
            risk_display = risk_display[risk_display["warehouse_id"] == selected_wh]

        n_crit = int((risk_display["risk_tier"] == "Critical").sum())
        n_high = int((risk_display["risk_tier"] == "High").sum())
        avg_cvar = risk_display["cvar_95"].mean()
        avg_ratio = risk_display["cvar_to_mean_ratio"].mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Critical-Tier SKUs", n_crit)
        c2.metric("High-Tier SKUs", n_high)
        c3.metric("Avg CVaR₉₅", f"${avg_cvar:,.0f}")
        c4.metric("Avg CVaR / E[Cost]", f"{avg_ratio:.1f}×", help="Tail amplification ratio. >2× = fat-tailed demand.")

        col_l, col_r = st.columns(2)
        with col_l:
            tier_counts = risk_display["risk_tier"].value_counts().reset_index()
            tier_counts.columns = ["Risk Tier", "Count"]
            fig_tier = px.pie(
                tier_counts, names="Risk Tier", values="Count",
                title="SKU-Warehouse Pairs by Risk Tier",
                color="Risk Tier",
                color_discrete_map={
                    "Critical": "#ff4b4b",
                    "High": "#ffa500",
                    "Medium": "#ffd700",
                    "Low": "#00cc88",
                },
            )
            st.plotly_chart(fig_tier, use_container_width=True)

        with col_r:
            fig_scatter = px.scatter(
                risk_display,
                x="mean_horizon_cost",
                y="cvar_95",
                color="risk_tier",
                size="stockout_probability",
                hover_data=["sku_id", "warehouse_id", "demand_distribution", "cvar_to_mean_ratio"],
                title="Expected Cost vs CVaR₉₅ (bubble size = stockout probability)",
                labels={
                    "mean_horizon_cost": "E[Cost] ($)",
                    "cvar_95": "CVaR₉₅ ($)",
                },
                color_discrete_map={
                    "Critical": "#ff4b4b",
                    "High": "#ffa500",
                    "Medium": "#ffd700",
                    "Low": "#00cc88",
                },
            )
            # 45° reference line (CVaR = E[cost], no tail risk)
            max_val = float(max(risk_display["cvar_95"].max(), risk_display["mean_horizon_cost"].max()))
            fig_scatter.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode="lines", line=dict(dash="dash", color="gray"),
                name="CVaR = E[Cost] (no tail)", showlegend=True,
            ))
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Top risky SKUs
        st.markdown("#### Highest CVaR₉₅ SKU-Warehouse Pairs")
        top_risk = risk_display.sort_values("cvar_95", ascending=False).head(20)
        st.dataframe(
            top_risk[[
                "sku_id", "warehouse_id", "risk_tier", "demand_distribution",
                "mean_horizon_cost", "var_95", "cvar_95",
                "cvar_to_mean_ratio", "stockout_probability",
            ]].style.format({
                "mean_horizon_cost": "${:,.0f}",
                "var_95": "${:,.0f}",
                "cvar_95": "${:,.0f}",
                "cvar_to_mean_ratio": "{:.2f}×",
                "stockout_probability": "{:.1%}",
            }),
            use_container_width=True,
        )

        if selected_sku and selected_sku in risk_display["sku_id"].values:
            st.markdown(f"#### {selected_sku} — Risk Distribution (all warehouses)")
            sku_risk = risk_display[risk_display["sku_id"] == selected_sku]
            fig_sku = go.Figure()
            for _, row in sku_risk.iterrows():
                # Reconstruct approximate cost distribution from statistics
                mu, sigma = row["mean_horizon_cost"], row["std_horizon_cost"]
                x_range = np.linspace(max(0, mu - 4 * sigma), mu + 4 * sigma, 300)
                from scipy.stats import norm
                y_pdf = norm.pdf(x_range, mu, max(sigma, 1.0))
                fig_sku.add_trace(go.Scatter(
                    x=x_range, y=y_pdf, fill="tozeroy",
                    name=row["warehouse_id"],
                    mode="lines",
                ))
                fig_sku.add_vline(
                    x=row["cvar_95"],
                    line_dash="dash",
                    annotation_text=f"CVaR₉₅ {row['warehouse_id']}",
                )
            fig_sku.update_layout(
                title=f"{selected_sku} — Approximate Cost Distribution per Warehouse",
                xaxis_title="8-week Horizon Cost ($)",
                yaxis_title="Density",
            )
            st.plotly_chart(fig_sku, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Azure Platform
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("### Azure Cloud Platform")

    c1, c2, c3 = st.columns(3)
    with c1:
        status = "✅ Connected" if az.get("storage_connected") else ("⚙ Configured" if az.get("configured") else "⬜ Not configured")
        st.metric("ADLS Gen2", status)
        if az.get("storage_account") != "—":
            st.caption(az["storage_account"])
    with c2:
        ml_status = "✅ Connected" if az.get("ml_configured") else "⬜ Not configured"
        st.metric("Azure ML", ml_status)
        if az.get("ml_workspace") != "—":
            st.caption(az["ml_workspace"])
    with c3:
        kv_status = "✅ Connected" if az.get("keyvault_configured") else "⬜ Not configured"
        st.metric("Key Vault", kv_status)

    st.markdown("---")

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### Medallion Architecture (Bronze → Silver → Gold)")
        st.markdown("""
| Layer | Path | Contents |
|-------|------|----------|
| **Bronze** | `bronze/raw/supply_chain/year=YYYY/month=MM/` | Raw immutable data, date-partitioned |
| **Silver** | `silver/processed/supply_chain/` | Cleaned, validated, feature-engineered |
| **Gold** | `gold/serving/supply_chain/` | ML outputs, model artifacts, manifests |
        """)
        st.markdown("""
**Why ADLS Gen2 (not plain Blob Storage)**
- Hierarchical Namespace = real O(1) directory rename/move
- POSIX ACLs per directory (column-level security)
- Native Databricks Delta / Azure Synapse integration
- Partition pruning for Spark queries (`year=YYYY/month=MM/`)
        """)

    with col_r:
        st.markdown("#### Authentication — DefaultAzureCredential Chain")
        st.markdown("""
```
DefaultAzureCredential tries in order:
  1. EnvironmentCredential    ← CI/CD (GitHub Actions, Azure DevOps)
  2. WorkloadIdentityCredential ← AKS pods
  3. ManagedIdentityCredential  ← Azure VMs / App Service (zero secrets)
  4. AzureCliCredential         ← Local dev (az login)
  5. InteractiveBrowserCredential
```
**Zero hardcoded credentials.** Same code runs locally, in CI, and in production.
        """)
        st.markdown("#### Azure ML Model Registry")
        st.markdown("""
```
Model lifecycle:
  Train → None → Staging → Production → Archived

MLflow run logged per pipeline execution:
  - Hyperparameters (n_estimators, max_depth, ...)
  - Metrics (MAPE, RMSE, conformal coverage, CVaR, ...)
  - Artifacts (model .pkl, feature importances)
  - Tags (pipeline version, azure_mode, run timestamp)
```
        """)

    st.markdown("---")
    st.markdown("#### Quick Setup")
    st.code("""# 1. Login to Azure
az login
az account set --subscription <your-subscription-id>

# 2. Provision all resources (one-time)
bash infra/provision.sh

# 3. Copy the printed env vars into .env
# (storage account, Key Vault URL, ML workspace, etc.)

# 4. Run pipeline with Azure upload
python run_pipeline.py --azure

# 5. View runs in Azure ML Studio
# https://ml.azure.com → Experiments → supply-chain-demand-forecasting
""", language="bash")

    if not az.get("configured"):
        st.info(
            "Azure is not configured yet. "
            "Set `AZURE_STORAGE_ACCOUNT_NAME` in your `.env` file (copy `.env.example`). "
            "Run `bash infra/provision.sh` to create all Azure resources automatically."
        )

    # MLflow runs table
    st.markdown("#### MLflow Experiment Runs (local)")
    try:
        import mlflow
        from pathlib import Path as P
        mlruns_path = P(__file__).parent / "mlruns"
        if mlruns_path.exists():
            mlflow.set_tracking_uri(f"file://{mlruns_path}")
            exp = mlflow.get_experiment_by_name("supply-chain-demand-forecasting")
            if exp:
                runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], max_results=10)
                if not runs.empty:
                    display_cols = [c for c in runs.columns if any(
                        k in c for k in ["run_id", "status", "start_time", "metrics.", "params.model_type"]
                    )]
                    st.dataframe(runs[display_cols].head(10), use_container_width=True)
                else:
                    st.caption("No runs logged yet. Run the pipeline to generate MLflow runs.")
            else:
                st.caption("No experiment found. Run pipeline first.")
        else:
            st.caption("No local MLflow runs yet.")
    except Exception:
        st.caption("MLflow runs unavailable.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — API Reference
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown("### REST API Reference")
    st.info(
        "A production **FastAPI** service exposes all intelligence layers as REST endpoints. "
        "Start it with:  `uvicorn api.main:app --reload --port 8000`\n\n"
        "Interactive docs available at `http://localhost:8000/docs`"
    )

    endpoints = [
        ("GET", "/health", "Service health + data freshness check"),
        ("GET", "/forecast/{sku_id}", "Point forecast + conformal intervals, filterable by region"),
        ("GET", "/forecast/calibration/metrics", "Mondrian calibration q̂ and empirical coverage"),
        ("GET", "/inventory/recommendations", "EOQ-based recommendations, filterable by risk/category"),
        ("GET", "/inventory/mip/{sku_id}", "MIP-optimal order schedule for a SKU"),
        ("GET", "/resilience/scores", "Per-node resilience scores and SPOF flags"),
        ("GET", "/resilience/spof", "Single points of failure only"),
        ("GET", "/network/edges", "Supply network edge list with capacities"),
        ("GET", "/risk/summary", "CVaR risk summary, filterable by tier"),
        ("GET", "/risk/{sku_id}/{warehouse_id}", "Full Monte Carlo risk profile for one pair"),
    ]

    for method, path, desc in endpoints:
        col_m, col_p, col_d = st.columns([1, 3, 5])
        col_m.markdown(f"**`{method}`**")
        col_p.code(path)
        col_d.markdown(desc)

    st.markdown("---")
    st.markdown("#### Example — Get forecast with intervals for SKU-001")
    st.code("curl http://localhost:8000/forecast/SKU-001?region_id=R-WEST", language="bash")

    st.markdown("#### Example — Get all Critical-tier risk SKUs")
    st.code("curl http://localhost:8000/risk/summary?risk_tier=Critical", language="bash")

    st.markdown("#### Example — Single points of failure")
    st.code("curl http://localhost:8000/resilience/spof", language="bash")

    st.markdown("---")
    st.markdown(
        "**Tech stack summary:**\n"
        "- **Forecast**: Random Forest + Mondrian Split Conformal Prediction (finite-sample coverage)\n"
        "- **Inventory**: EOQ baseline + Multi-Period Lot-Sizing MIP (PuLP/CBC solver)\n"
        "- **Resilience**: NetworkX directed flow graph, max-flow resilience scoring\n"
        "- **Risk**: Monte Carlo (10k scenarios), Negative Binomial / Poisson demand, CVaR₉₅\n"
        "- **API**: FastAPI + Pydantic v2 response models, async CORS-enabled\n"
        "- **Dashboard**: Streamlit + Plotly with interactive filters"
    )

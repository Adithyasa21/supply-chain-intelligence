"""
Supply Chain Network Resilience Analysis via NetworkX.

Models the supply network as a directed flow graph and measures how much
throughput capacity is lost when any single node fails.

Graph topology
--------------
  [Supplier nodes]  →  [Warehouse nodes]  →  [Region nodes]
       SUP-xx             WH-xx                  R-xx

Edge capacities:
  Supplier → Warehouse:  avg weekly SKU volume flowing through that path
  Warehouse → Region:    weekly_capacity_units × routing share to that region

Resilience metrics computed
---------------------------
1. Betweenness centrality
   Fraction of all shortest paths that pass through each node.
   High centrality → routing bottleneck.

2. Max-flow resilience  R(v)
   R(v) = MaxFlow(G − {v}) / MaxFlow(G)
   Fraction of total supply capacity retained when node v is removed.
   R(v) < 0.50  →  Single Point of Failure (SPOF).

3. Concentration risk
   What share of total flow passes through each warehouse.

4. Redundancy score
   For each (supplier, region) pair: number of distinct warehouse paths
   that can carry demand. Low redundancy = brittle lane.

Senior-engineer notes
---------------------
Using NetworkX's max_flow (push-relabel) is conceptually equivalent to
solving a min-cost flow LP for capacity planning. The graph abstraction
naturally extends to multi-echelon networks (add DC tier, retail tier).
"""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


_SPOF_THRESHOLD = 0.50   # R(v) below this → flagged as single point of failure


def _build_graph(
    warehouses: pd.DataFrame,
    regions: pd.DataFrame,
    shipping: pd.DataFrame,
    lead_times: pd.DataFrame,
    orders: pd.DataFrame,
) -> nx.DiGraph:
    G = nx.DiGraph()

    # ── Warehouse nodes ──────────────────────────────────────────────────────
    for _, wh in warehouses.iterrows():
        G.add_node(
            wh["warehouse_id"],
            node_type="warehouse",
            weekly_capacity=float(wh["weekly_capacity_units"]),
            label=wh["warehouse_name"],
        )

    # ── Region nodes ─────────────────────────────────────────────────────────
    for _, r in regions.iterrows():
        G.add_node(
            r["region_id"],
            node_type="region",
            market_size=float(r["market_size_index"]),
            label=r["region_name"],
        )

    # ── Supplier nodes (derived from lead-time table) ────────────────────────
    supplier_ids = lead_times["primary_supplier_id"].unique()
    for sup in supplier_ids:
        G.add_node(sup, node_type="supplier", label=sup)

    # ── Supplier → Warehouse edges ───────────────────────────────────────────
    # Capacity proxy: number of unique SKUs × avg weekly demand per lane
    weekly_per_wh = (
        orders.groupby("warehouse_id", as_index=False)["order_units"]
        .mean()
        .rename(columns={"order_units": "avg_weekly_units"})
    )
    for _, lt in lead_times.iterrows():
        sup = lt["primary_supplier_id"]
        wh = lt["warehouse_id"]
        rel = 1.0 / max(1.0, float(lt["avg_supplier_lead_time_days"]))  # shorter lead = more reliable
        wh_vol = weekly_per_wh[weekly_per_wh["warehouse_id"] == wh]["avg_weekly_units"]
        cap = float(wh_vol.iloc[0]) if len(wh_vol) else 1.0
        if G.has_edge(sup, wh):
            G[sup][wh]["capacity"] += cap
            G[sup][wh]["n_skus"] += 1
        else:
            G.add_edge(sup, wh, capacity=cap, reliability=rel, n_skus=1, edge_type="supply")

    # ── Warehouse → Region edges ─────────────────────────────────────────────
    # Capacity: warehouse_capacity × fraction of orders going to that region
    region_share = (
        orders.groupby(["warehouse_id", "region_id"], as_index=False)["order_units"].sum()
    )
    wh_totals = region_share.groupby("warehouse_id")["order_units"].transform("sum")
    region_share["share"] = region_share["order_units"] / wh_totals.replace(0, 1)

    wh_cap = warehouses.set_index("warehouse_id")["weekly_capacity_units"]
    for _, row in region_share.iterrows():
        wh = row["warehouse_id"]
        r = row["region_id"]
        cap = float(wh_cap.get(wh, 1.0)) * float(row["share"])
        cost_row = shipping[(shipping["warehouse_id"] == wh) & (shipping["region_id"] == r)]
        transit = float(cost_row["base_transit_days"].iloc[0]) if len(cost_row) else 5.0
        G.add_edge(wh, r, capacity=cap, transit_days=transit, share=float(row["share"]), edge_type="fulfillment")

    # ── Virtual super-source and super-sink for max-flow ────────────────────
    G.add_node("_SOURCE", node_type="virtual")
    G.add_node("_SINK", node_type="virtual")

    total_cap = sum(
        G.nodes[s].get("weekly_capacity", 1e9)
        for s in supplier_ids
    )
    for sup in supplier_ids:
        G.add_edge("_SOURCE", sup, capacity=total_cap)

    for r in regions["region_id"]:
        G.add_edge(r, "_SINK", capacity=total_cap)

    return G


def _max_flow(G: nx.DiGraph) -> float:
    try:
        flow_val, _ = nx.maximum_flow(G, "_SOURCE", "_SINK", capacity="capacity")
        return float(flow_val)
    except Exception:
        return 0.0


def run_resilience_analysis(root: Path) -> None:
    processed = root / "data" / "processed"
    raw = root / "data" / "raw"

    warehouses = pd.read_csv(raw / "warehouses.csv")
    regions = pd.read_csv(raw / "regions.csv")
    shipping = pd.read_csv(raw / "shipping_cost_matrix.csv")
    lead_times = pd.read_csv(raw / "supplier_lead_times.csv")
    orders = pd.read_csv(raw / "orders.csv")

    G = _build_graph(warehouses, regions, shipping, lead_times, orders)

    # ── Baseline max flow ────────────────────────────────────────────────────
    baseline_flow = _max_flow(G)

    # ── Betweenness centrality (exclude virtual nodes) ───────────────────────
    real_nodes = [n for n in G.nodes if not str(n).startswith("_")]
    centrality = nx.betweenness_centrality(G, normalized=True, weight=None)

    # ── Node failure resilience scores ───────────────────────────────────────
    rows = []
    for node in real_nodes:
        node_type = G.nodes[node].get("node_type", "unknown")

        # Remove node and all its edges
        G_removed = G.copy()
        G_removed.remove_node(node)
        # Reconnect virtual source/sink if removed node broke them
        if node in G_removed:
            pass
        if "_SOURCE" not in G_removed or "_SINK" not in G_removed:
            residual_flow = 0.0
        else:
            residual_flow = _max_flow(G_removed)

        resilience = residual_flow / max(baseline_flow, 1.0)
        flow_loss = baseline_flow - residual_flow
        is_spof = resilience < _SPOF_THRESHOLD

        rows.append({
            "node_id": node,
            "node_type": node_type,
            "label": G.nodes[node].get("label", node),
            "betweenness_centrality": round(centrality.get(node, 0.0), 6),
            "residual_flow_if_removed": round(residual_flow, 2),
            "baseline_flow": round(baseline_flow, 2),
            "resilience_score": round(resilience, 4),
            "flow_loss_if_removed": round(flow_loss, 2),
            "flow_loss_pct": round((1 - resilience) * 100, 1),
            "is_single_point_of_failure": is_spof,
        })

    resilience_df = pd.DataFrame(rows).sort_values("resilience_score")

    # ── Concentration risk per warehouse ─────────────────────────────────────
    wh_flows = []
    for wh_id in warehouses["warehouse_id"]:
        in_flow = sum(
            G[u][wh_id].get("capacity", 0)
            for u in G.predecessors(wh_id)
            if not str(u).startswith("_")
        )
        wh_flows.append({"warehouse_id": wh_id, "inbound_capacity": round(in_flow, 2)})
    wh_flow_df = pd.DataFrame(wh_flows)
    total_wh_cap = wh_flow_df["inbound_capacity"].sum()
    wh_flow_df["concentration_pct"] = (
        wh_flow_df["inbound_capacity"] / max(total_wh_cap, 1) * 100
    ).round(1)

    # ── Redundancy: unique warehouse paths per (supplier, region) ────────────
    redundancy_rows = []
    supplier_nodes = [n for n in real_nodes if G.nodes[n].get("node_type") == "supplier"]
    region_nodes = [n for n in real_nodes if G.nodes[n].get("node_type") == "region"]
    for sup in supplier_nodes:
        for reg in region_nodes:
            paths = list(nx.all_simple_paths(G, sup, reg, cutoff=3))
            redundancy_rows.append({
                "supplier_id": sup,
                "region_id": reg,
                "n_paths": len(paths),
                "redundancy_level": "High" if len(paths) >= 3 else ("Medium" if len(paths) >= 2 else "Low"),
            })
    redundancy_df = pd.DataFrame(redundancy_rows)

    # ── Save outputs ─────────────────────────────────────────────────────────
    resilience_df.to_csv(processed / "resilience_scores.csv", index=False)
    wh_flow_df.to_csv(processed / "warehouse_concentration.csv", index=False)
    redundancy_df.to_csv(processed / "network_redundancy.csv", index=False)

    # Summary stats for edges (used by dashboard)
    edge_rows = [
        {
            "source": u,
            "target": v,
            "capacity": round(d.get("capacity", 0), 2),
            "edge_type": d.get("edge_type", ""),
        }
        for u, v, d in G.edges(data=True)
        if not str(u).startswith("_") and not str(v).startswith("_")
    ]
    pd.DataFrame(edge_rows).to_csv(processed / "network_edges.csv", index=False)

    n_spof = int(resilience_df["is_single_point_of_failure"].sum())
    min_r = resilience_df["resilience_score"].min()
    print(
        f"  Resilience analysis: {len(real_nodes)} nodes, "
        f"baseline flow={baseline_flow:,.0f}, "
        f"{n_spof} single points of failure, "
        f"min resilience score={min_r:.2%}."
    )

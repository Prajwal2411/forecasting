from __future__ import annotations

import math
from typing import Dict, List, Tuple, Any

import pandas as pd


def _round_lrm(values: List[float], total_target: int | None = None) -> List[int]:
    vals = [float(v) for v in values]
    if total_target is None:
        total_target = int(round(sum(vals)))
    floors = [int(math.floor(v)) for v in vals]
    residual = int(max(0, total_target - sum(floors)))
    rema = [v - f for v, f in zip(vals, floors)]
    order = sorted(range(len(rema)), key=lambda i: rema[i], reverse=True)
    for i in range(min(residual, len(order))):
        floors[order[i]] += 1
    return floors


def _scale_role_totals(df: pd.DataFrame, desired_team_size: int | None) -> Dict[str, int]:
    role_sums = df.groupby("role_family", as_index=False)["demand"].sum()
    weights = {row["role_family"]: float(row["demand"]) for _, row in role_sums.iterrows()}
    if not desired_team_size:
        return {k: int(round(v)) for k, v in weights.items()}
    total_w = sum(weights.values()) or 1.0
    scaled = {k: (v / total_w) * float(desired_team_size) for k, v in weights.items()}
    floors = {k: int(float(v)) for k, v in scaled.items()}
    residual = int(desired_team_size) - sum(floors.values())
    order = sorted(scaled.items(), key=lambda kv: (float(kv[1]) - int(float(kv[1]))), reverse=True)
    for i in range(max(0, residual)):
        floors[order[i][0]] += 1
    return floors


def build_plan(forecast_table: List[dict], desired_team_size: int | None = None) -> Dict[str, Any]:
    """Compute once and reuse: role totals, level mix, quarterly targets, internal share, and costs.

    Returns a dict with keys:
      - role_totals: Dict[str,int]
      - exp_mix: pd.DataFrame with columns [Role, Senior, Mid, Junior]
      - quarterly_counts: pd.DataFrame with [quarter, role_family, level, demand_count, internal_count, external_count]
      - totals: Dict with total demand, internal, external, cost
    """
    df = pd.DataFrame(forecast_table or [])
    if df.empty:
        return {
            "role_totals": {},
            "exp_mix": pd.DataFrame(columns=["Role", "Senior", "Mid", "Junior"]),
            "quarterly_counts": pd.DataFrame(columns=["quarter","role_family","level","demand_count","internal_count","external_count"]),
            "totals": {"forecast_total_demand": 0.0, "internal_total": 0.0, "external_total": 0.0, "total_cost": 0.0},
        }

    # Role totals (scaled to desired team if specified)
    role_totals = _scale_role_totals(df, desired_team_size)

    # Level mix per role (bucket Lead/Principal into Senior)
    buck = {"Lead": "Senior", "Principal": "Senior", "Senior": "Senior", "Mid": "Mid", "Junior": "Junior"}
    df["bucket"] = df["level"].map(lambda x: buck.get(str(x), "Mid"))
    mix = df.groupby(["role_family", "bucket"], as_index=False)["demand"].sum()
    pv = mix.pivot(index="role_family", columns="bucket", values="demand").fillna(0.0)
    for col in ["Senior", "Mid", "Junior"]:
        if col not in pv.columns:
            pv[col] = 0.0
    pv = pv[["Senior", "Mid", "Junior"]]

    # Convert to integer per role according to role_totals using LRM
    rows_out = []
    for role_name, row in pv.iterrows():
        target = int(role_totals.get(str(role_name), int(round(row.sum()))))
        vals = [float(row.get("Senior", 0.0)), float(row.get("Mid", 0.0)), float(row.get("Junior", 0.0))]
        Sr, Md, Jr = _round_lrm(vals, total_target=target)
        # Pod ratio guard: Junior <= Senior; if violated, move to Mid
        if Jr > Sr:
            move = Jr - Sr
            Jr -= move; Md += move
        rows_out.append({"Role": str(role_name), "Senior": Sr, "Mid": Md, "Junior": Jr})
    exp_mix = pd.DataFrame(rows_out)

    # Quarter distribution per (role, bucket) proportional to demand shares
    df_g = df.groupby(["quarter", "role_family", "bucket"], as_index=False).agg({
        "demand": "sum",
        "internal_fulfilled": "sum",
    })

    # Build integer targets per (role,bucket)
    target_by_rl: Dict[Tuple[str, str], int] = {}
    for _, row in exp_mix.iterrows():
        r = row["Role"]
        for b in ("Senior", "Mid", "Junior"):
            target_by_rl[(r, b)] = int(row.get(b, 0))

    rows_q: List[dict] = []
    for (role, bucket), T in target_by_rl.items():
        if T <= 0:
            continue
        sub = df_g[(df_g["role_family"] == role) & (df_g["bucket"] == bucket)]
        if sub.empty:
            continue
        weights = sub["demand"].astype(float).tolist()
        if sum(weights) <= 0:
            weights = [1.0] * len(weights)
        d_counts = _round_lrm([w / sum(weights) * T for w in weights], total_target=T)
        # Internal share per group
        tot_dem = float(sub["demand"].sum())
        tot_int = float(sub["internal_fulfilled"].sum())
        int_total = int(round(T * (tot_int / tot_dem))) if tot_dem > 0 else 0
        iw = sub["internal_fulfilled"].astype(float).tolist()
        if sum(iw) <= 0:
            iw = weights[:]
        i_counts = _round_lrm([w / sum(iw) * int_total for w in iw], total_target=int_total)
        # Cap internal by demand per quarter
        for i in range(len(i_counts)):
            i_counts[i] = min(i_counts[i], d_counts[i])
        for i, (_, rsub) in enumerate(sub.iterrows()):
            dc = int(d_counts[i]); ic = int(i_counts[i]); ec = dc - ic
            if dc <= 0:
                continue
            rows_q.append({
                "quarter": rsub["quarter"],
                "role_family": role,
                "level": bucket,
                "demand_count": dc,
                "internal_count": ic,
                "external_count": ec,
            })
    quarterly_counts = pd.DataFrame(rows_q).sort_values(["quarter", "role_family", "level"]).reset_index(drop=True)

    # Totals
    totals = {
        "forecast_total_demand": float(df["demand"].sum()) if "demand" in df.columns else 0.0,
        "internal_total": float(df.get("internal_fulfilled", pd.Series()).sum()) if "internal_fulfilled" in df.columns else 0.0,
        "external_total": float(df.get("external_hires", pd.Series()).sum()) if "external_hires" in df.columns else 0.0,
        "total_cost": float(df.get("cost_estimated", pd.Series()).sum()) if "cost_estimated" in df.columns else 0.0,
    }

    return {
        "role_totals": role_totals,
        "exp_mix": exp_mix,
        "quarterly_counts": quarterly_counts,
        "totals": totals,
    }


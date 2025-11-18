from typing import Dict, List, Tuple

import math
import pandas as pd


def round_lrm(vals: List[float], total_target: int | None = None) -> List[int]:
    v = [float(x) for x in vals]
    if total_target is None:
        total_target = int(round(sum(v)))
    floors = [int(math.floor(x)) for x in v]
    residual = int(max(0, total_target - sum(floors)))
    rem = [x - f for x, f in zip(v, floors)]
    order = sorted(range(len(rem)), key=lambda i: rem[i], reverse=True)
    for i in range(min(residual, len(order))):
        floors[order[i]] += 1
    return floors


def aggregate_forecast_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate tech rows to Quarter x Role x Level and compute demand‑weighted/filled‑weighted salary.
    Expects columns: quarter, role_family, level, demand, (avg_salary), (internal_fulfilled), (external_hires), (cost_estimated).
    """
    if df.empty:
        return df
    group_cols = ["quarter", "role_family", "level"]
    if "avg_salary" in df.columns:
        d = pd.to_numeric(df.get("demand"), errors="coerce").fillna(0.0)
        if {"internal_fulfilled", "external_hires"}.issubset(df.columns):
            i = pd.to_numeric(df.get("internal_fulfilled"), errors="coerce").fillna(0.0)
            e = pd.to_numeric(df.get("external_hires"), errors="coerce").fillna(0.0)
            w = i + e
            if float(w.sum() or 0.0) <= 0.0:
                w = d
        else:
            w = d
        df = df.copy()
        df["_w"] = w
        df["_wsal"] = pd.to_numeric(df.get("avg_salary"), errors="coerce").fillna(0.0) * w
    agg_dict: Dict[str, str] = {"demand": "sum"}
    for c in ("internal_fulfilled", "external_hires", "cost_estimated", "_w", "_wsal"):
        if c in df.columns:
            agg_dict[c] = "sum"
    g = df.groupby(group_cols, as_index=False).agg(agg_dict)
    if set(["_w", "_wsal"]).issubset(g.columns) and (g["_w"] > 0).any():
        g["avg_salary"] = (g["_wsal"] / g["_w"]).round(2)
        g = g.drop(columns=["_w", "_wsal"], errors="ignore")
    # Reorder columns for readability if present
    cols = [c for c in [
        "quarter", "role_family", "level", "demand", "avg_salary", "internal_fulfilled", "external_hires", "cost_estimated"
    ] if c in g.columns]
    return g[cols]


def compute_quarterly_hiring_counts(forecast_table: List[dict], desired_team_size: int | None) -> pd.DataFrame:
    """Compute integer counts per quarter, role, level.
    Total team size applies across the whole project, not per quarter.
    Filters out zero-demand rows.
    """
    df_f = pd.DataFrame(forecast_table)
    required = {"quarter", "role_family", "level", "demand"}
    if df_f.empty or not required.issubset(df_f.columns):
        return pd.DataFrame()

    # Normalize levels to buckets
    buck = {"Lead": "Senior", "Principal": "Senior", "Senior": "Senior", "Mid": "Mid", "Junior": "Junior"}
    df_f["role_display"] = df_f["role_family"].astype(str)
    df_f["exp_bucket"] = df_f["level"].map(lambda x: buck.get(str(x), "Mid"))

    # Whole‑window role totals (FTE) and scale to desired team size if provided
    role_sums = df_f.groupby("role_display", as_index=False)["demand"].sum()
    role_sums = role_sums.rename(columns={"role_display": "Role", "demand": "count"})
    if desired_team_size:
        weights = {row["Role"]: float(row["count"]) for _, row in role_sums.iterrows()}
        total_w = sum(weights.values()) or 1.0
        scaled = {k: (v / total_w) * float(desired_team_size) for k, v in weights.items()}
        floors = {k: int(float(v)) for k, v in scaled.items()}
        residual = int(desired_team_size) - sum(floors.values())
        order = sorted(scaled.items(), key=lambda kv: (float(kv[1]) - int(float(kv[1]))), reverse=True)
        for i in range(max(0, residual)):
            floors[order[i][0]] += 1
        role_totals = {k: floors[k] for k in floors.keys()}
    else:
        role_totals = {row["Role"]: int(round(float(row["count"]))) for _, row in role_sums.iterrows()}

    # Mix by bucket across the window
    mix = df_f.groupby(["role_display", "exp_bucket"], as_index=False)["demand"].sum()
    pivot = mix.pivot(index="role_display", columns="exp_bucket", values="demand").fillna(0.0)
    for col in ["Senior", "Mid", "Junior"]:
        if col not in pivot.columns:
            pivot[col] = 0.0
    pivot = pivot[["Senior", "Mid", "Junior"]]

    # Targets per Role x Level (integers)
    target_by_rl: Dict[Tuple[str, str], int] = {}
    for role_name, row in pivot.iterrows():
        target = int(role_totals.get(str(role_name), int(round(row.sum()))))
        scaled = [float(row.get("Senior", 0.0)), float(row.get("Mid", 0.0)), float(row.get("Junior", 0.0))]
        total_v = sum(scaled) or 1.0
        scaled = [v / total_v * target for v in scaled]
        Sr, Md, Jr = round_lrm(scaled, total_target=target)
        target_by_rl[(str(role_name), "Senior")] = Sr
        target_by_rl[(str(role_name), "Mid")] = Md
        target_by_rl[(str(role_name), "Junior")] = Jr

    # Quarter weights and internal shares
    g_q = df_f.groupby(["quarter", "role_display", "exp_bucket"], as_index=False).agg({
        "demand": "sum",
        "internal_fulfilled": "sum",
    })

    rows_out: List[Dict] = []
    for (role_name, lvl), total_T in target_by_rl.items():
        sub = g_q[(g_q["role_display"] == role_name) & (g_q["exp_bucket"] == lvl)].copy()
        if sub.empty or total_T == 0:
            continue
        w = [float(x) for x in sub["demand"].tolist()]
        if sum(w) <= 0:
            w = [1.0] * len(w)
        d_counts = round_lrm([x / sum(w) * total_T for x in w], total_target=total_T)
        tot_dem = float(sub["demand"].sum())
        tot_int = float(sub["internal_fulfilled"].sum())
        int_total_target = max(0, min(total_T, int(round(total_T * (tot_int / tot_dem))) if tot_dem > 0 else 0))
        iw = [float(x) for x in sub["internal_fulfilled"].tolist()]
        if sum(iw) <= 0:
            iw = w[:]
        i_counts = round_lrm([x / sum(iw) * int_total_target for x in iw], total_target=int_total_target)
        overflow = 0
        for i in range(len(i_counts)):
            if i_counts[i] > d_counts[i]:
                overflow += i_counts[i] - d_counts[i]
                i_counts[i] = d_counts[i]
        if overflow > 0:
            headroom_idx = [i for i in range(len(d_counts)) if d_counts[i] - i_counts[i] > 0]
            for i in headroom_idx:
                take = min(overflow, d_counts[i] - i_counts[i])
                i_counts[i] += take
                overflow -= take
                if overflow <= 0:
                    break
        for i, (_, row) in enumerate(sub.iterrows()):
            dc = int(d_counts[i])
            ic = int(i_counts[i])
            ec = dc - ic
            if dc <= 0:
                continue
            rows_out.append({
                "quarter": row["quarter"],
                "role_family": role_name,
                "level": lvl,
                "demand_count": dc,
                "internal_count": ic,
                "external_count": ec,
            })

    df_counts = pd.DataFrame(rows_out)
    if not df_counts.empty:
        df_counts = df_counts[df_counts["demand_count"] > 0]
        df_counts = df_counts.sort_values(["quarter", "role_family", "level"])  # type: ignore
    return df_counts


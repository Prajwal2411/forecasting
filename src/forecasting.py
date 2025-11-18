import csv
import os
from typing import Any, Dict, List, Tuple
try:
    import pandas as _pd  # optional for Parquet fast path
except Exception:
    _pd = None
from collections import defaultdict


def read_market_trends(path: str) -> List[dict]:
    """Load market trends with Parquet fast path when available.
    If a sibling .parquet file exists and pandas+engine are available, prefer it.
    Otherwise fall back to CSV.
    """
    # Attempt Parquet twin
    try:
        parquet_path = path[:-4] + ".parquet" if path.lower().endswith(".csv") else None
        if parquet_path and os.path.exists(parquet_path) and _pd is not None:
            df = _pd.read_parquet(parquet_path)
            return df.to_dict(orient="records")  # type: ignore
    except Exception:
        pass
    # Fallback CSV
    rows: List[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def parse_quarter(q: str) -> Tuple[int, int]:
    # YYYYQn
    return int(q[:4]), int(q[-1])


def in_quarter(year: int, quarter: int, qstr: str) -> bool:
    y, q = parse_quarter(qstr)
    return y == year and q == quarter


def baseline_headcount_forecast(
    jd: Dict[str, Any],
    market_trends_csv: str,
    start_quarter: str,
    end_quarter: str,
    region: str | None,
) -> List[dict]:
    # Simple baseline: demand signal averaged across JD technologies in domain and region (or global)
    mt = read_market_trends(market_trends_csv)
    y0, q0 = parse_quarter(start_quarter)
    y1, q1 = parse_quarter(end_quarter)

    # enumerate quarters inclusively
    quarters = []
    y, q = y0, q0
    while (y < y1) or (y == y1 and q <= q1):
        quarters.append((y, q))
        q = q + 1
        if q > 4:
            q = 1
            y += 1

    tech_items = jd.get("tech_requirements", []) or []
    techs = [t["technology"] for t in tech_items] or []
    prio_w = {t["technology"]: (1.5 if float(t.get("priority", 2)) >= 3 else 1.0) for t in tech_items}
    domain = jd.get("domain")
    roles_needed = jd.get("roles_needed", [])

    # Compute role mix weights normalized by counts and level_mix
    mix: Dict[Tuple[str, str], float] = defaultdict(float)  # (role, level) -> weight
    for r in roles_needed:
        rf = r["role_family"]
        cnt = max(1, int(r.get("count", 1)))
        lvl_mix = r.get("level_mix", {})
        for lvl, w in lvl_mix.items():
            mix[(rf, lvl)] += cnt * float(w)

    # Normalize mix sum to 1.0 per role then overall
    total_w = sum(mix.values()) or 1.0
    for k in list(mix.keys()):
        mix[k] /= total_w

    # Role/level salary multipliers
    role_mult = {
        "PM": 1.1,
        "Architect": 1.3,
        "Backend": 1.0,
        "Frontend": 0.95,
        "FullStack": 1.05,
        "QA": 0.8,
        "DevOps": 1.15,
        "DataEng": 1.2,
        "DataSci": 1.25,
        "MLE": 1.35,
        "Mobile": 1.0,
        "UI/UX": 0.9,
        "Support": 0.75,
    }
    level_mult = {"Junior": 0.7, "Mid": 1.0, "Senior": 1.4, "Lead": 1.7, "Principal": 2.1}

    # For each quarter, compute a base FTE volume from average demand_index over techs
    out: List[dict] = []
    for (yy, qq) in quarters:
        filt = [r for r in mt if int(r["year"]) == yy and int(r["quarter"]) == qq];
        if domain:
            filt = [r for r in filt if r["domain"] == domain]
        if region:
            region_rows = [r for r in filt if (r["region"] == region)]
            if region_rows:
                filt = region_rows
        # Subset by techs, but if nothing matches, fall back to domain-level average
        filt_for_weights = filt
        if techs:
            filt_by_tech = [r for r in filt if r["technology"] in techs]
            if filt_by_tech:
                filt = filt_by_tech
                filt_for_weights = filt_by_tech
        if not filt:
            continue
        # Weighted by tech priority when applicable
        if techs and filt_for_weights:
            num = 0.0; den = 0.0; num_sal = 0.0
            for r in filt_for_weights:
                w = float(prio_w.get(r["technology"], 1.0))
                num += w * float(r["demand_index"])
                num_sal += w * float(r.get("median_salary", 0.0))
                den += w
            if den > 0:
                avg_demand = num / den
                avg_salary = num_sal / den
            else:
                avg_demand = sum(float(r["demand_index"]) for r in filt) / len(filt)
                avg_salary = sum(float(r.get("median_salary", 0.0)) for r in filt) / len(filt)
        else:
            avg_demand = sum(float(r["demand_index"]) for r in filt) / len(filt)
            avg_salary = sum(float(r.get("median_salary", 0.0)) for r in filt) / len(filt)
        # Map demand to FTE: simple scaling where 50 -> 8 FTE baseline
        base_fte = max(1.0, avg_demand / 6.0)
        # Distribute demand across technologies instead of only the first tech
        if techs:
            # Compute per-tech weight from priority; normalize
            wsum = sum(prio_w.get(t, 1.0) for t in techs) or 1.0
            tech_weights = {t: prio_w.get(t, 1.0) / wsum for t in techs}
        else:
            tech_weights = {"General": 1.0}

        # Salary by tech for the quarter (fallback to avg_salary)
        sal_by_tech = {}
        for t in tech_weights.keys():
            rows_t = [r for r in mt if int(r["year"]) == yy and int(r["quarter"]) == qq and r["technology"] == (t if t != "General" else r["technology"])];
            if domain:
                rows_t = [r for r in rows_t if r["domain"] == domain]
            if region:
                rows_t = [r for r in rows_t if r.get("region") == region] or rows_t
            if rows_t:
                sal_by_tech[t] = sum(float(r.get("median_salary", 0.0)) for r in rows_t) / len(rows_t)
            else:
                sal_by_tech[t] = avg_salary

        for (role, level), w in mix.items():
            role_fte = base_fte * w
            for t, tw in tech_weights.items():
                demand = role_fte * tw
                # Apply role/level multipliers to salary
                rm = float(role_mult.get(role, 1.0))
                lm = float(level_mult.get(level, 1.0))
                base_sal = sal_by_tech.get(t, avg_salary)
                adj_salary = max(25000.0, base_sal * rm * lm)
                out.append({
                    "quarter": f"{yy}Q{qq}",
                    "role_family": role,
                    "level": level,
                    "technology": t,
                    "demand": round(demand, 2),
                    "avg_salary": round(adj_salary, 2),
                })
    return out

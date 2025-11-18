from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple
import csv
import os
try:
    import pandas as _pd
except Exception:
    _pd = None


def _read_employees(path: str) -> Dict[int, dict]:
    try:
        pq = path[:-4] + ".parquet" if path.lower().endswith(".csv") else None
        if pq and os.path.exists(pq) and _pd is not None:
            df = _pd.read_parquet(pq)
            return {int(r["employee_id"]): r for r in df.to_dict(orient="records")}
    except Exception:
        pass
    out: Dict[int, dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[int(row["employee_id"])] = row
    return out


def _read_assignments(path: str) -> List[dict]:
    try:
        pq = path[:-4] + ".parquet" if path.lower().endswith(".csv") else None
        if pq and os.path.exists(pq) and _pd is not None:
            df = _pd.read_parquet(pq)
            return df.to_dict(orient="records")  # type: ignore
    except Exception:
        pass
    rows: List[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        rows.extend(csv.DictReader(f))
    return rows


def _release_quarter(date_str: str) -> str:
    from datetime import datetime
    d = datetime.fromisoformat(date_str)
    q = (d.month - 1) // 3 + 1
    return f"{d.year}Q{q}"


def optimize_allocation_lp(
    forecast: List[dict],
    employees_csv: str,
    assignments_csv: str,
    budget_ceiling: float | None,
    region: str | None,
    currency_factor: float | None = 1.0,
) -> Tuple[List[dict], List[dict], List[dict]]:
    try:
        import pulp
    except Exception as e:
        raise RuntimeError("PuLP is not installed. Please install 'pulp' to use LP optimizer.") from e

    employees = _read_employees(employees_csv)
    assignments = _read_assignments(assignments_csv)

    # Build availability map: per quarter, employee has capacity 1.0 if release falls in or before that quarter
    # Simple approach: release quarter availability for exactly that quarter.
    avail_by_q: Dict[str, set[int]] = defaultdict(set)
    for a in assignments:
        q = _release_quarter(a["assigned_end_date_planned"])
        avail_by_q[q].add(int(a["employee_id"]))

    # Group forecast rows by quarter and filter candidates by role
    # Decision variables:
    #  - x_ext[i] >= 0 continuous: external FTE for forecast row i
    #  - x_int[e, i] in [0, 1]: fraction of internal FTE for employee e to row i (only if available and role matches)
    # Constraints:
    #  - For each row i: sum_e x_int[e,i] + x_ext[i] >= demand_i
    #  - For each (q, e): sum_i_in_q x_int[e,i] <= 1.0
    #  - Budget: sum(cost_int* x_int + cost_ext * x_ext) <= budget
    # Objective: maximize total coverage sum_i (x_ext[i] + sum_e x_int[e,i])

    # Formulate as a cost minimization with exact coverage per row
    prob = pulp.LpProblem("AllocationLP", pulp.LpMinimize)

    # Index forecast rows
    I = list(range(len(forecast)))
    # Candidate internals per row
    candidates: Dict[int, List[int]] = {}
    def _has_tech(eid: int, tech: str) -> bool:
        if not tech or tech == "General":
            return True
        e = employees[eid]
        skills = set()
        pt = (e.get("primary_tech") or "").strip()
        if pt:
            skills.add(pt)
        secs = (e.get("secondary_techs") or "").split(",")
        skills |= {s.strip() for s in secs if s.strip()}
        return tech in skills

    for i, row in enumerate(forecast):
        q = row["quarter"]
        role = row["role_family"]
        tech = row.get("technology")
        role_pool = [e for e in avail_by_q.get(q, set()) if employees[e]["role_family"] == role]
        tech_pool = [e for e in role_pool if _has_tech(e, tech)] if role_pool else []
        candidates[i] = tech_pool if tech_pool else role_pool

    # Variables
    x_ext = {i: pulp.LpVariable(f"x_ext_{i}", lowBound=0) for i in I}
    x_int = {}
    for i in I:
        for e in candidates[i]:
            x_int[(e, i)] = pulp.LpVariable(f"x_int_{e}_{i}", lowBound=0, upBound=1)

    # Costs: internal cost per quarter approximated by CTC/4; external by a flat rate per 1.0 FTE
    def ext_cost(row: dict) -> float:
        # Use forecast-provided avg_salary (annual) if present, else fallback
        try:
            sal = float(row.get("avg_salary", 0.0))
            if sal > 0:
                return sal / 4.0
        except Exception:
            pass
        return 20000.0

    def emp_cost_per_q(eid: int) -> float:
        ctc = float(employees[eid]["cost_to_company_per_year"])
        fx = float(currency_factor or 1.0)
        return (ctc * fx) / 4.0

    # Coverage constraints: exactly meet demand per row to avoid unbounded solutions
    for i, row in enumerate(forecast):
        demand = float(row["demand"])
        prob += x_ext[i] + pulp.lpSum(x_int[(e, i)] for e in candidates[i]) == demand

    # Capacity constraints per employee per quarter
    # Build rows per quarter per employee participation
    rows_by_q: Dict[str, List[int]] = defaultdict(list)
    for i, row in enumerate(forecast):
        rows_by_q[row["quarter"]].append(i)
    for q, eids in avail_by_q.items():
        iq = rows_by_q.get(q, [])
        for e in eids:
            involved = [x_int[(e, i)] for i in iq if (e, i) in x_int]
            if involved:
                prob += pulp.lpSum(involved) <= 1.0

    # Budget constraint (optional)
    total_cost = pulp.lpSum(
        [x_ext[i] * ext_cost(forecast[i]) for i in I]
        + [x_int[(e, i)] * emp_cost_per_q(e) for (e, i) in x_int.keys()]
    )
    if budget_ceiling is not None:
        prob += total_cost <= budget_ceiling

    # Objective: minimize total cost while meeting exact demand
    prob += total_cost

    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Extract solution and build tables
    forecast_table: List[dict] = []
    hiring_plan: List[dict] = []
    reassignment_plan: List[dict] = []

    # Internal fill by row
    internal_by_i: Dict[int, float] = defaultdict(float)
    for (e, i), var in x_int.items():
        val = float(var.value() or 0.0)
        if val > 1e-6:
            internal_by_i[i] += val
            reassignment_plan.append({
                "employee_id": e,
                "quarter": forecast[i]["quarter"],
                "role_family": forecast[i]["role_family"],
                "level": employees[e]["level"],
                "technology": forecast[i]["technology"],
                "fte": round(val, 2),
            })

    # Aggregate external hires per quarter/role/level (keep fractional to preserve small needs)
    ext_map: Dict[Tuple[str, str, str], float] = defaultdict(float)
    for i in I:
        ext = float(x_ext[i].value() or 0.0)
        if ext > 1e-6:
            key = (forecast[i]["quarter"], forecast[i]["role_family"], forecast[i]["level"])
            ext_map[key] += ext
    for (q, role, level), v in ext_map.items():
        cnt = round(float(v), 2)
        if cnt <= 0.0:
            continue
        hiring_plan.append({
            "quarter": q,
            "role_family": role,
            "level": level,
            "external_hires": cnt,
        })

    # Cost estimate
    # Per-row cost estimates
    cost_by_i: Dict[int, float] = defaultdict(float)
    for i in I:
        ext = float(x_ext[i].value() or 0.0)
        if ext > 1e-6:
            cost_by_i[i] += ext * ext_cost(forecast[i])
    for (e, i), var in x_int.items():
        val = float(var.value() or 0.0)
        if val > 1e-6:
            cost_by_i[i] += val * emp_cost_per_q(e)

    for i, row in enumerate(forecast):
        forecast_table.append({
            **row,
            "internal_fulfilled": round(internal_by_i.get(i, 0.0), 2),
            "external_hires": round(float(x_ext[i].value() or 0.0), 2),
            "cost_estimated": round(cost_by_i.get(i, 0.0), 2),
        })

    return forecast_table, hiring_plan, reassignment_plan

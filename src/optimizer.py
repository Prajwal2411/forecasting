from collections import defaultdict
from typing import Any, Dict, List, Tuple
import csv
import os
try:
    import pandas as _pd  # optional for Parquet fast path
except Exception:
    _pd = None


def read_employees(path: str) -> Dict[int, dict]:
    # Parquet fast path
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


def read_assignments(path: str) -> List[dict]:
    # Parquet fast path
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


def parse_quarter(q: str) -> Tuple[int, int]:
    return int(q[:4]), int(q[-1])


def plan_allocation(
    forecast: List[dict],
    employees_csv: str,
    assignments_csv: str,
    budget_ceiling: float | None,
    region: str | None,
    median_salary_lookup: Dict[Tuple[str, str], float] | None = None,
) -> Tuple[List[dict], List[dict], List[dict]]:
    # Simple greedy allocator: fill internal first from employees whose assignments end near quarter, then external within budget
    employees = read_employees(employees_csv)
    assignments = read_assignments(assignments_csv)

    # Build capacity using the same ±1 month window around each forecast quarter start
    from datetime import datetime, timedelta

    def quarter_start(qs: str) -> datetime:
        y, qn = int(qs[:4]), int(qs[-1])
        return datetime(y, 3 * (qn - 1) + 1, 1)

    quarters_in_forecast = sorted({str(r.get("quarter")) for r in forecast if r.get("quarter")})

    # Capacity pool per quarter: eid -> remaining FTE (start as 1.0)
    capacity: Dict[str, Dict[int, float]] = defaultdict(dict)
    for a in assignments:
        try:
            eid = int(a["employee_id"])
            d = datetime.fromisoformat(a["assigned_end_date_planned"]).replace(hour=0, minute=0, second=0, microsecond=0)
        except Exception:
            continue
        for qstr in quarters_in_forecast:
            qs = quarter_start(qstr)
            if (qs - timedelta(days=30)) <= d <= (qs + timedelta(days=30)):
                capacity[qstr][eid] = 1.0

    # Track budget usage (includes internal reassignment cost and external hiring cost)
    budget_used = 0.0
    forecast_table: List[dict] = []
    hiring_plan: List[dict] = []
    # Aggregate reassignment entries by (employee, quarter, role, level)
    reassignment_agg: Dict[Tuple[int, str, str, str], Dict[str, Any]] = {}

    # Helper: technology match between forecast row and employee skills
    def _has_tech(e: dict, tech: str) -> bool:
        if not tech or tech == "General":
            return True
        skills = set()
        pt = (e.get("primary_tech") or "").strip()
        if pt:
            skills.add(pt)
        secs = (e.get("secondary_techs") or "").split(",")
        skills |= {s.strip() for s in secs if s.strip()}
        return tech in skills

    # Helper: bucket employee/forecast levels to comparable categories
    def _level_bucket(lvl: str | None) -> str:
        x = (lvl or "").strip()
        if x in ("Lead", "Principal"):
            return "Senior"
        if x in ("Senior", "Mid", "Junior"):
            return x
        # Default bucket for unknowns
        return "Mid"

    # Greedy per row of forecast
    for row in forecast:
        q = row["quarter"]
        role = row["role_family"]
        level = row["level"]
        tech = row["technology"]
        needed = float(row["demand"])  # may be fractional; we'll round to 0.5 FTE chunks
        internal = 0.0
        external = 0.0
        row_cost = 0.0

        # Internal fills using capacity[q] based on primary tech/role/level in 0.5 increments
        chunk = 0.5
        capq = capacity.get(q, {})
        # Multi-pass assignment: (tech+level) -> (tech only) -> (level only) -> (role only)
        # Sort candidates: prefer matching role
        cand_ids = sorted(capq.keys(), key=lambda eid: 0 if employees[eid]["role_family"] == role else 1)
        for tech_pass in (True, False):
            for level_pass in (True, False):
                for eid in cand_ids:
                    if internal >= needed:
                        break
                    if capq.get(eid, 0.0) <= 0:
                        continue
                    e = employees[eid]
                    if e["role_family"] != role:
                        continue
                    if tech_pass and not _has_tech(e, tech):
                        continue
                    if level_pass and _level_bucket(e.get("level")) != _level_bucket(level):
                        continue
                    take = min(chunk, capq[eid], needed - internal)
                    if take <= 0:
                        continue
                    # Estimate internal cost per quarter for this employee
                try:
                    emp_ctc = float(e.get("cost_to_company_per_year", 0.0))
                except Exception:
                    emp_ctc = 0.0
                emp_q_cost = max(0.0, emp_ctc / 4.0)
                # Respect budget ceiling when adding internal capacity
                if budget_ceiling is not None and emp_q_cost > 0:
                    remaining_budget = max(0.0, float(budget_ceiling) - float(budget_used))
                    max_afford = remaining_budget / emp_q_cost if emp_q_cost > 0 else take
                    if max_afford <= 0:
                        continue
                    # Clamp take to what we can afford now
                    take = min(take, max_afford)
                    # Snap to 2 decimals to avoid floating drift
                    take = round(take, 2)
                    if take <= 0:
                        continue
                    capq[eid] -= take
                    internal += take
                    # Accrue internal cost into row cost and budget
                    if emp_q_cost > 0:
                        row_cost += emp_q_cost * take
                        budget_used += emp_q_cost * take
                    # Record reassignment at the planned (bucketed) level from forecast, not the employee's raw level
                    planned_lvl = _level_bucket(level)
                    key = (eid, q, role, planned_lvl)
                    entry = reassignment_agg.get(key)
                    if not entry:
                        entry = {
                            "employee_id": eid,
                            "quarter": q,
                            "role_family": role,
                            "level": planned_lvl,
                            "technology": set([tech]) if tech else set(),
                            "fte": 0.0,
                        }
                        reassignment_agg[key] = entry
                    else:
                        if tech:
                            entry["technology"].add(tech)
                    entry["fte"] = round(float(entry["fte"]) + float(take), 2)
            # break if filled
            if internal >= needed:
                break

        remaining = max(0.0, needed - internal)
        if remaining > 0:
            # External hires in 0.5 increments within budget
            while remaining > 0:
                est_cost = None
                if median_salary_lookup:
                    # Try detailed keys first, then fallback
                    est_cost = (
                        median_salary_lookup.get((role, level, tech, q))
                        or median_salary_lookup.get((role, tech, q))
                        or median_salary_lookup.get((tech, q))
                    )
                if est_cost is None:
                    # Fallback: derive from forecasted annual avg_salary when available (per quarter per 1.0 FTE)
                    try:
                        est_cost = float(row.get("avg_salary", 80000.0)) / 4.0
                    except Exception:
                        est_cost = 20000.0
                take = min(chunk, remaining)
                if budget_ceiling is not None:
                    remaining_budget = max(0.0, float(budget_ceiling) - float(budget_used))
                    max_afford = remaining_budget / float(est_cost) if est_cost > 0 else take
                    if max_afford <= 0:
                        break
                    take = min(take, max_afford)
                    take = round(take, 2)
                    if take <= 0:
                        break
                budget_used += est_cost * take
                row_cost += est_cost * take
                external += take
                remaining -= take

        forecast_table.append({
            **row,
            "internal_fulfilled": round(internal, 2),
            "external_hires": round(external, 2),
            "cost_estimated": round(row_cost, 2),
        })

        if external > 0:
            hiring_plan.append({
                "quarter": q,
                "role_family": role,
                "level": level,
                "technology": tech,
                "external_hires": external,
            })

    # Convert aggregated map to list; sort by quarter then employee_id
    reassignment_plan: List[dict] = []
    for (_eid, _q, _role, _lvl), entry in reassignment_agg.items():
        techs = ", ".join(sorted(list(entry.get("technology") or []))) if isinstance(entry.get("technology"), set) else entry.get("technology")
        reassignment_plan.append({
            "employee_id": entry["employee_id"],
            "quarter": entry["quarter"],
            "role_family": entry["role_family"],
            "level": entry["level"],
            "technology": techs,
            "fte": entry["fte"],
        })
    reassignment_plan.sort(key=lambda r: (r.get("quarter", ""), int(r.get("employee_id", 0))))

    # Aggregate external hiring plan (keep fractional to preserve small needs)
    ext_by_key = defaultdict(float)
    for r in hiring_plan:
        key = (r.get("quarter"), r.get("role_family"), r.get("level"))
        try:
            ext_by_key[key] += float(r.get("external_hires", 0.0))
        except Exception:
            pass

    # Keep fractional hires to reflect distribution across quarters
    hiring_plan_out: List[dict] = []
    for (q, role, level), v in ext_by_key.items():
        cnt = round(float(v), 2)
        if cnt <= 0.0:
            continue
        hiring_plan_out.append({
            "quarter": q,
            "role_family": role,
            "level": level,
            "external_hires": cnt,
        })

    return forecast_table, hiring_plan_out, reassignment_plan

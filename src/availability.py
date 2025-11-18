import csv
import os
from typing import List
try:
    import pandas as _pd  # optional
except Exception:
    _pd = None
from datetime import datetime, timedelta
from typing import Dict, List


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
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def parse_date(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _quarter_str(d: datetime) -> str:
    q = (d.month - 1) // 3 + 1
    return f"{d.year}Q{q}"


def compute_release_notes(assignments_csv: str, quarter: str) -> List[dict]:
    # quarter format YYYYQn
    year = int(quarter[:4])
    q = int(quarter[-1])
    q_start = datetime(year, 3 * (q - 1) + 1, 1)
    q_end = datetime(year, 3 * q, 1)
    window_start = q_start - timedelta(days=30)
    window_end = q_start + timedelta(days=30)

    rows = read_assignments(assignments_csv)
    # Deduplicate by employee: keep the nearest release to the quarter start within the window
    by_emp: Dict[int, dict] = {}
    for row in rows:
        try:
            eid = int(row["employee_id"])
            rel = parse_date(row["assigned_end_date_planned"]).replace(hour=0, minute=0, second=0, microsecond=0)
        except Exception:
            continue
        if not (window_start <= rel <= window_end):
            continue
        dist = abs((rel - q_start).days)
        cur = by_emp.get(eid)
        if (cur is None) or (dist < cur["_dist"]):
            by_emp[eid] = {
                "employee_id": eid,
                "release_date": rel.date().isoformat(),
                "release_quarter": _quarter_str(rel),
                "role": row.get("role_on_project"),
                "tech_stack": row.get("tech_stack"),
                "notice_period_days": int(row.get("notice_period_days", 0) or 0),
                "_dist": dist,
            }
    # Drop helper key
    out = []
    for v in by_emp.values():
        v.pop("_dist", None)
        out.append(v)
    # Sort by release date
    out.sort(key=lambda r: r.get("release_date", ""))
    return out

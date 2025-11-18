
import math
import os
from pathlib import Path
import streamlit as st

from typing import Any

import pandas as pd
import numpy as np
try:
    import plotly.express as px
except Exception:
    px = None

try:
    import plotly.graph_objects as go
except Exception:
    go = None

from src.analytics_plan import build_plan, _round_lrm
from src.utils.ontology import (
    ROLE_TAXONOMY,
    get_hiring_rule_targets,
    get_hiring_rule_tolerance_percent,
    role_level_mix_default,
)
ROOT = Path(__file__).resolve().parents[1]
def run_pipeline(
    jd_text: str,
    budget: float,
    start_q: str,
    end_q: str,
    region: str,
    use_lp: bool,
    jd_tech_text: str | None = None,
    desired_team_size: int | None = None,
    profit_margin_pct: float = 0.0,
):
    from src.data_synthesis import DATA_DIR, gen_market_trends, gen_employees_projects_assignments
    from src.availability import compute_release_notes
    from src.jd_parser import parse_jd
    from src.llm_match import score_match
    from src.forecasting import baseline_headcount_forecast
    from src.optimizer import plan_allocation as greedy
    mt_path = os.path.join(DATA_DIR, "market_trends_quarterly.csv")
    emp_path = os.path.join(DATA_DIR, "employees.csv")
    proj_path = os.path.join(DATA_DIR, "projects.csv")
    asg_path = os.path.join(DATA_DIR, "assignments.csv")
    # Ensure market trends cover the requested end year; regenerate if needed
    end_year_req = None
    try:
        end_year_req = int(end_q[:4]) if end_q and "Q" in end_q else None
    except Exception:
        end_year_req = None
    if not os.path.exists(mt_path):
        try:
            gen_market_trends(end_year=end_year_req)
        except TypeError:
            # Backward compatibility with older signature
            gen_market_trends()
    else:
        # Check max year in existing file; regenerate if insufficient
        try:
            import pandas as _pd
            _df_mt = _pd.read_csv(mt_path, usecols=["year"], nrows=100000)
            max_year = int(_df_mt["year"].max()) if not _df_mt.empty else 0
        except Exception:
            max_year = 0
        if end_year_req and end_year_req > max_year:
            try:
                gen_market_trends(end_year=end_year_req)
            except TypeError:
                gen_market_trends()
    if not (os.path.exists(emp_path) and os.path.exists(proj_path) and os.path.exists(asg_path)):
        gen_employees_projects_assignments()
    parsed = parse_jd(jd_text, tech_section=jd_tech_text)
    # Derive canonical role allocations before forecasting to keep tables in sync
    tech_rows = parsed.get("tech_requirements", []) or []
    parsed_roles = parsed.get("roles_needed", []) or []
    hiring_targets = get_hiring_rule_targets()
    hiring_tolerance = get_hiring_rule_tolerance_percent()
    roles_simple, allocation_source, final_alloc, team_size = _compute_role_allocation(
        tech_rows,
        parsed_roles,
        int(desired_team_size) if desired_team_size else None,
        hiring_targets,
        hiring_tolerance,
    )
    derived_info = {
        "roles_simple": roles_simple,
        "allocation_source": allocation_source,
        "final_alloc": final_alloc,
        "team_size": team_size,
    }
    derived_role_rows = []
    for row in roles_simple:
        lvl_mix = role_level_mix_default(str(row["role_family"]))
        derived_role_rows.append({"role_family": row["role_family"], "count": row["count"], "level_mix": lvl_mix})
    parsed["_derived_roles"] = derived_info
    if derived_role_rows:
        parsed["_original_roles_needed"] = parsed_roles
        parsed["roles_needed"] = derived_role_rows
    # Use India as default when region is '(any)' or not selected for salary lookups
    region_eff = region if region and region not in ("(any)", "(select region)") else "India"
    forecast = baseline_headcount_forecast(parsed, mt_path, start_q, end_q, region_eff)
    # If a desired team size is provided, scale the forecasted demand PER QUARTER to match it
    # Scale per-quarter so total demand each quarter equals the desired team size (constant team across project)
    try:
        if desired_team_size and forecast:
            by_q = {}
            for r in forecast:
                q = r.get("quarter")
                by_q[q] = by_q.get(q, 0.0) + float(r.get("demand", 0.0))
            for r in forecast:
                q = r.get("quarter")
                total_q = by_q.get(q, 0.0)
                if total_q > 0:
                    scale = float(desired_team_size) / total_q
                    r["demand"] = round(float(r.get("demand", 0.0)) * scale, 2)
    except Exception:
        pass
    # Release suggestions: employees releasing within ï¿½1 month of start_q and matching JD role families
    release_notes = compute_release_notes(asg_path, start_q)
    jd_roles = set(r.get("role_family") for r in parsed.get("roles_needed", []) if r.get("role_family"))
    jd_tr = parsed.get("tech_requirements", []) or []
    jd_techs = set(t.get("technology") for t in jd_tr if t.get("technology"))
    prio_map = {t.get("technology"): int(t.get("priority", 2)) for t in jd_tr}
    # Load employee names/levels for display
    emp_df = pd.read_csv(emp_path)
    emp_map = emp_df.set_index("employee_id")["name"].to_dict()
    emp_level_map = emp_df.set_index("employee_id")["level"].to_dict()
    rel_rows = []
    for rn in release_notes:
        rf = str(rn.get("role", "")).split("-", 1)[0]
        if rf not in jd_roles:
            continue
        # Tech stack match: check assignment tech_stack, or employee primary/secondary
        stack = (rn.get("tech_stack") or "").split(",")
        stack = {s.strip() for s in stack if s}
        eid = int(rn["employee_id"])
        emp_row = emp_df.loc[emp_df["employee_id"] == eid]
        emp_primary = set([str(emp_row["primary_tech"].iloc[0])]) if not emp_row.empty else set()
        emp_secondaries = set(str(emp_row["secondary_techs"].iloc[0]).split(",")) if not emp_row.empty else set()
        tech_match = (jd_techs & stack) or (jd_techs & emp_primary) or (jd_techs & emp_secondaries)
        if not jd_techs or tech_match:
            # Determine top priority matched
            match_set = (jd_techs & stack) or (jd_techs & emp_primary) or (jd_techs & emp_secondaries)
            top_prio = max([prio_map.get(t, 2) for t in match_set], default=2)
            # LLM/embedding-based similarity score (fallback to tag overlap)
            jd_text_for_score = "; ".join(sorted(jd_techs))
            cand_text = ", ".join(sorted(list(stack | emp_primary | emp_secondaries)))
            score = 0.0
            try:
                score = score_match(jd_text_for_score, cand_text, jd_tags=jd_techs, cand_tags=(stack | emp_primary | emp_secondaries))
            except Exception:
                score = float(len(match_set)) / max(1, len(jd_techs))
            rel_rows.append({
                "employee_id": eid,
                "name": emp_map.get(eid),
                "role_family": rf,
                "level": emp_level_map.get(eid),
                "release_date": rn.get("release_date"),
                "release_month": rn.get("release_date", "")[:7],
                "notice_period_days": rn.get("notice_period_days"),
                "tech_stack": rn.get("tech_stack"),
                "priority_match": top_prio,
                "match_score": round(float(score), 4),
            })
    # Rank suggestions: higher priority first, earlier release first
    def _date_key(d):
        try:
            return pd.to_datetime(d)
        except Exception:
            return pd.NaT
    rel_rows.sort(key=lambda r: (-(r.get("match_score", 0.0) or 0.0), -(r.get("priority_match", 2) or 2), _date_key(r.get("release_date"))))
    total_demand_all = sum(float(r.get("demand", 0.0)) for r in (forecast or []))
    eff_budget = None if budget is None else float(budget) * (1.0 - float(profit_margin_pct or 0.0)/100.0)
    if use_lp:
        try:
            from src.optimizer_lp import optimize_allocation_lp
            forecast_table, hiring_plan, reassignment_plan = optimize_allocation_lp(
                forecast, emp_path, asg_path, eff_budget, region_eff,
            )
        except Exception as e:
            st.warning(f"LP optimizer unavailable: {e}. Falling back to greedy.")
            # Build salary lookup per (technology, quarter) using forecast avg_salary (annual -> per quarter)
            sal_lookup = { (r.get("technology"), r.get("quarter")): float(r.get("avg_salary", 0.0))/4.0 for r in forecast }
            forecast_table, hiring_plan, reassignment_plan = greedy(
                  forecast, emp_path, asg_path, eff_budget, region_eff, sal_lookup,
              )
        # If LP succeeds, keep LP results (no greedy overwrite)
    return parsed, forecast_table, hiring_plan, reassignment_plan, rel_rows
st.set_page_config(page_title="Human Resource Forecasting", layout="wide")
st.title("Human Resource Forecasting")
st.sidebar.header("Inputs")
# Currency selection (display + input currency)
currency_options = {
    "USD": {"symbol": "$", "fx": 1.0},
    "EUR": {"symbol": "ï¿½", "fx": 0.92},
    "INR": {"symbol": "?", "fx": 83.0},
}
sel_currency = st.sidebar.selectbox("Currency", list(currency_options.keys()), index=0)
_c = currency_options[sel_currency]
CURRENCY_SYMBOL = _c["symbol"]
# Factor means: 1 base unit ~= fx selected currency units
# Internally we compute in base units; inputs are interpreted in selected currency, so we divide by fx
CURRENCY_FX = float(_c["fx"]) or 1.0
# Start with blank inputs; validate on submit
budget_text = st.sidebar.text_input("Budget", value="", placeholder="e.g., 750000")
start_q_raw = st.sidebar.text_input("Start Quarter (Qn-YYYY)", value="", placeholder="e.g., Q1-2025")
end_q_raw = st.sidebar.text_input("End Quarter (Qn-YYYY)", value="", placeholder="e.g., Q3-2025")
region = st.sidebar.selectbox("Region", ["(select region)", "India", "NA", "EU", "APAC", "(any)"])
desired_team_size = st.sidebar.number_input("Team Size", min_value=1, max_value=500, value=8, step=1)
min_profit_margin = st.sidebar.number_input("Min Profit Margin (%)", min_value=0, max_value=80, value=10, step=1)
budget = None
if budget_text.strip():
    try:
        # Interpret entered budget in the selected currency, convert to internal base
        _val = float(budget_text.replace(",", ""))
        budget = _val / CURRENCY_FX
    except Exception:
        budget = None
# Hide LP toggle in UI; keep optimizer enabled by default
use_lp = True
# Toggle charts visibility
show_charts = st.sidebar.checkbox("Show Charts", value=True)
if st.sidebar.button("Regenerate Data"):
    try:
        from src.data_synthesis import DATA_DIR, gen_market_trends, gen_employees_projects_assignments
        import glob
        # Remove existing CSV/Parquet to force fresh generation
        for pat in ("*.csv", "*.parquet"):
            for p in glob.glob(os.path.join(DATA_DIR, pat)):
                try:
                    os.remove(p)
                except Exception:
                    pass
        gen_market_trends()
        gen_employees_projects_assignments()
        st.success("Synthetic data regenerated. Rerun forecast to use refreshed data.")
    except Exception as e:
        st.warning(f"Could not regenerate data: {e}")
default_jd_path = ROOT / "sample_jd.txt"
# Start blank; user can paste JD. Optionally show sample via expander.
jd_text = st.text_area("Job Description (what you need)", value="", height=220, placeholder="Paste project/role description here...")
jd_tech_text = st.text_area(
    "Tech Stack & Experience (strict format)",
    value="",
    height=140,
    placeholder="One role per line. Examples:\nFrontend: React 3y, Angular 2y, JavaScript 4 years\nDataEng (3): Spark 3y, Kafka 2y, Airflow 2y\nDevOps x2: Kubernetes 2y, AWS 2y",
)
with st.expander("Need an example JD? Show sample"):
    if default_jd_path.exists():
        st.code(default_jd_path.read_text(encoding="utf-8"))
# Helper to render tables with currency formatting (no index column)
def show_df(df_like):
    df = pd.DataFrame(df_like)
    # Standardize column headers: replace underscores with spaces, Title Case words
    def _pretty_col(c):
        s = str(c).replace("_", " ")
        s = " ".join([w for w in s.split(" ") if w != ""])  # collapse extra spaces
        return s.title()
    df.columns = [_pretty_col(c) for c in df.columns]
    # Currency-format money-related columns for display
    MONEY_KEYS = ("Salary", "Budget", "Cost", "Profit")
    money_cols = [c for c in df.columns if any(k in c for k in MONEY_KEYS)]
    for c in money_cols:
        # Convert to numeric, scale for display currency, and format with symbol
        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.notna().any():
            scaled = vals * CURRENCY_FX
            df[c] = scaled.map(lambda x: f"{CURRENCY_SYMBOL}{x:,.2f}" if pd.notna(x) else "")
    # Reset index to avoid showing it; hide index in display
    df = df.reset_index(drop=True)
    st.dataframe(df, use_container_width=True, hide_index=True)


def _apply_hiring_rule_constraints(
    alloc: dict[str, int], rule_targets: dict[str, int], tolerance_pct: float
) -> dict[str, int]:
    if not rule_targets or not alloc:
        return alloc
    capped: dict[str, int] = {}
    tol = max(0.0, float(tolerance_pct or 0.0))
    for role, count in alloc.items():
        target = rule_targets.get(role)
        if target is None:
            capped[role] = int(count)
            continue
        lower = int(math.floor(target * (1.0 - tol / 100.0)))
        upper = int(math.ceil(target * (1.0 + tol / 100.0)))
        if lower < 0:
            lower = 0
        if upper < lower:
            upper = lower
        capped[role] = min(max(int(round(count)), lower), upper)
    return {r: c for r, c in capped.items() if c > 0}


def _derive_roles_from_taxonomy(
    jd_skills: set[str],
    taxonomy: dict[str, dict],
    hiring_targets: dict[str, int],
    desired_total: int | None,
) -> dict[str, int]:
    if not jd_skills:
        return {}
    skill_lookup = {s.lower() for s in jd_skills if s}
    matches: dict[str, int] = {}
    for role, info in taxonomy.items():
        role_skills = {s.lower() for s in info.get("skills", []) if s}
        overlap = len(role_skills & skill_lookup)
        if overlap > 0:
            matches[role] = overlap
    if not matches:
        return {}
    total_overlap = sum(matches.values())
    derived: dict[str, int] = {}
    for role, overlap in matches.items():
        base_target = hiring_targets.get(role, 1)
        weight = overlap / total_overlap if total_overlap > 0 else 1.0 / len(matches)
        derived[role] = max(1, int(round(base_target * weight)))
    if desired_total and sum(derived.values()) > 0:
        scale = desired_total / sum(derived.values())
        derived = {role: max(1, int(round(count * scale))) for role, count in derived.items()}
    return derived


def _scale_alloc_to_target(alloc: dict[str, int], target: int) -> dict[str, int]:
    if target <= 0 or not alloc:
        return alloc
    total = sum(alloc.values())
    if total <= 0:
        return alloc
    items = list(alloc.items())
    ratios = [count / total * target for _, count in items]
    ints = _round_lrm(ratios, total_target=target)
    scaled: dict[str, int] = {}
    for (role, _), val in zip(items, ints):
        scaled[role] = max(0, int(val))
    return scaled


def _compute_role_allocation(
    tech_rows: list[dict[str, Any]],
    parsed_roles: list[dict[str, Any]],
    desired_team: int | None,
    hiring_targets: dict[str, int],
    tolerance_pct: float,
) -> tuple[list[dict[str, int]], str | None, dict[str, int], int | None]:
    roles_simple: list[dict[str, int]] = []
    allocation_source: str | None = None
    jd_skills = {t.get("technology") for t in (tech_rows or []) if t.get("technology")}
    role_alloc_map: dict[str, int] = {}
    if jd_skills:
        role_alloc_map = _derive_roles_from_taxonomy(
            jd_skills,
            ROLE_TAXONOMY,
            hiring_targets,
            int(desired_team) if desired_team else None,
        )
        if role_alloc_map:
            allocation_source = "tech-taxonomy"
    if not role_alloc_map and parsed_roles:
        allocation_source = "parsed"
        role_alloc_map = {
            str(r.get("role_family")): int(r.get("count", 0))
            for r in parsed_roles
            if int(r.get("count", 0)) > 0
        }
    if not role_alloc_map and hiring_targets:
        allocation_source = "hiring_rule"
        role_alloc_map = {role: int(count) for role, count in hiring_targets.items() if int(count) > 0}
    constrained_map = _apply_hiring_rule_constraints(role_alloc_map, hiring_targets, tolerance_pct)
    final_alloc: dict[str, int] = {}
    if constrained_map:
        total_target = (
            int(desired_team) if desired_team and desired_team > 0 else sum(constrained_map.values())
        )
        if total_target > 0:
            final_alloc = _scale_alloc_to_target(constrained_map, total_target)
        else:
            final_alloc = constrained_map
    roles_simple = [
        {"role_family": role, "count": count}
        for role, count in sorted(final_alloc.items())
        if count > 0
    ]
    team_size = sum(final_alloc.values()) if final_alloc else (int(desired_team) if desired_team else None)
    return roles_simple, allocation_source, final_alloc, team_size
# Quarter helpers
import re
q_pattern = re.compile(r"^\s*Q\s*([1-4])\s*[-]?\s*([12]\d{3})\s*$", re.I)
q_pattern_yy = re.compile(r"^\s*([12]\d{3})\s*Q\s*([1-4])\s*$", re.I)
def normalize_q_input(s: str):
    if not s:
        return None
    m = q_pattern.match(s)
    if m:
        return f"{m.group(2)}Q{m.group(1)}"
    m2 = q_pattern_yy.match(s)
    if m2:
        return f"{m2.group(1)}Q{m2.group(2)}"
    return None
def to_qn_dash(yyyyqn: str):
    if not yyyyqn or len(yyyyqn) < 6 or "Q" not in yyyyqn:
        return yyyyqn
    return f"Q{yyyyqn[-1]}-{yyyyqn[:4]}"
if st.button("Run Forecast"):
    missing = []
    if not jd_text.strip():
        missing.append("JD text")
    start_q_norm = normalize_q_input(start_q_raw.strip()) if start_q_raw else None
    end_q_norm = normalize_q_input(end_q_raw.strip()) if end_q_raw else None
    if not start_q_norm or not end_q_norm:
        try:
            from src.jd_parser import parse_jd as _parse_hint
            hint = _parse_hint(jd_text) if jd_text.strip() else {}
        except Exception:
            hint = {}
        start_q_norm = start_q_norm or hint.get("start_quarter")
        end_q_norm = end_q_norm or hint.get("end_quarter")
    if not start_q_norm:
        missing.append("Start Quarter (Qn-YYYY or include in JD)")
    if not end_q_norm:
        missing.append("End Quarter (Qn-YYYY or include in JD)")
    if budget is None:
        missing.append("Budget Ceiling")
    if missing:
        st.error("Please enter: " + ", ".join(missing))
        st.stop()
    else:
        parsed, forecast_table, hiring_plan, reassignment_plan, release_suggestions = run_pipeline(
            jd_text,
            budget,
            start_q_norm,
            end_q_norm,
            None if region in ("(any)", "(select region)") else region,
            use_lp,
            jd_tech_text=jd_tech_text if jd_tech_text.strip() else None,
            desired_team_size=int(desired_team_size) if desired_team_size else None,
        )
    tech_rows = parsed.get("tech_requirements", []) or []
    hiring_rule_targets = get_hiring_rule_targets()
    hiring_rule_tolerance = get_hiring_rule_tolerance_percent()
    derived_info = parsed.get("_derived_roles", {})
    roles_simple = derived_info.get("roles_simple", [])
    allocation_source = derived_info.get("allocation_source")
    final_alloc = derived_info.get("final_alloc") or {}
    active_team_size = derived_info.get("team_size")
    role_alloc_override = dict(final_alloc) if final_alloc else None
    # Parsed JD as tables
    st.subheader("Parsed JD - Summary")
    constraints = parsed.get("constraints", {}) or {}
    # Show user inputs for start/end/budget/region so they are always visible
    # Use display-friendly Qn-YYYY in summary
    display_team_size = (
        active_team_size if active_team_size is not None else (int(desired_team_size) if desired_team_size else None)
    )
    summary = {
        "project_title": parsed.get("project_title"),
        "domain": parsed.get("domain"),
        "start_quarter": to_qn_dash(start_q_norm),
        "end_quarter": to_qn_dash(end_q_norm),
        "budget_ceiling": budget,
        # Show exactly what the user selected for region
        "region": region,
        "desired_team_size": display_team_size,
    }
    with st.expander("Parsed JD - Summary"):
        show_df([summary])
    st.subheader("Tech Requirements")
    if tech_rows:
        with st.expander("Tech Requirements"):
            show_df(tech_rows)
            # Brief explanation of the Priority column
            st.caption(
                "Priority indicates emphasis by tech: 3 for Cloud/Data stacks (higher weight in demand), 2 for others."
            )
    else:
        st.info("No technology requirements parsed.")
    # Report objects for downloads
    forecast_display_df = None
    roles_download_df = None
    exp_mix_df = None
    hiring_plan_df = None
    reassignment_plan_df = None
    totals_summary_df = None
    release_df = None
    st.subheader("Roles Needed")
    if allocation_source == "hiring_rule":
        st.info("Using hiring_rule.yml defaults because no JD-driven roles were inferred.")
    if roles_simple:
        label = "Roles Needed (derived from JD tech stack)" if allocation_source == "tech-taxonomy" else "Roles Needed"
        with st.expander(label):
            show_df(roles_simple)
            try:
                roles_download_df = pd.DataFrame(roles_simple)
            except Exception:
                roles_download_df = None
            if allocation_source == "hiring_rule" and hiring_rule_tolerance:
                st.caption(f"Derived from hiring_rule.yml defaults. Tolerance +/-{hiring_rule_tolerance}%.")
    else:
        st.info("Could not infer roles from the JD.")
    
    # Experience Mix (by Role) - placed before Forecast Table, computed once via plan
    exp_mix_df = None
    try:
        plan_target_size = int(active_team_size) if active_team_size else None
        plan_obj = build_plan(forecast_table or [], plan_target_size)
        exp_mix_df = plan_obj.get("exp_mix")
    except Exception:
        exp_mix_df = None
    if exp_mix_df is not None and hasattr(exp_mix_df, 'empty') and not exp_mix_df.empty:
        st.subheader("Experience Mix (by Role)")
        with st.expander("Experience Mix (by Role)"):
            # Ensure Experience Mix sums to Team Size even if upstream plan misses scaling
            try:
                from src.analytics import round_lrm as _round_lrm
            except Exception:
                def _round_lrm(vals, total_target=None):
                    import math as _math
                    v = [float(x) for x in vals]
                    if total_target is None:
                        total_target = int(round(sum(v)))
                    floors = [int(_math.floor(x)) for x in v]
                    residual = int(max(0, total_target - sum(floors)))
                    if residual > 0 and len(v) > 0:
                        rem = [x - f for x, f in zip(v, floors)]
                        order = sorted(range(len(rem)), key=lambda i: rem[i], reverse=True)
                        i = 0
                        while residual > 0 and order:
                            idx = order[i % len(order)]
                            floors[idx] += 1
                            residual -= 1
                            i += 1
                    return floors
            try:
                df_em = pd.DataFrame(exp_mix_df).copy()
                if "role_alloc_override" in locals() and role_alloc_override:
                    existing_roles = {str(r) for r in df_em.get("Role", []) if pd.notna(r)}
                    missing_roles = [r for r in role_alloc_override.keys() if r not in existing_roles]
                    extra_rows = []
                    for role in missing_roles:
                        buckets = ["Senior", "Mid", "Junior"]
                        mix = role_level_mix_default(role)
                        weights = [float(mix.get(b, 0.0)) for b in buckets]
                        if sum(weights) == 0.0:
                            weights = [1.0, 1.0, 0.0]
                        scaled = [(w / sum(weights)) * float(role_alloc_override.get(role, 0)) for w in weights]
                        ints = _round_lrm(scaled, total_target=int(role_alloc_override.get(role, 0)))
                        Sr, Md, Jr = (ints + [0, 0, 0])[:3]
                        extra_rows.append({"Role": role, "Senior": Sr, "Mid": Md, "Junior": Jr})
                    if extra_rows:
                        df_em = pd.concat([df_em, pd.DataFrame(extra_rows)], ignore_index=True)
                cols = [c for c in df_em.columns if str(c) in ("Senior","Mid","Junior")]
                # Apply per-role targets from hiring rules if available
                try:
                    if 'role_alloc_override' in locals() and role_alloc_override and cols:
                        df_em_roles = df_em.copy()
                        for i in range(len(df_em_roles)):
                            try:
                                rname = str(df_em_roles.at[i, 'Role'])
                            except Exception:
                                continue
                            if rname == 'Total':
                                continue
                            if rname not in {str(k) for k in (role_alloc_override or {}).keys()}:
                                continue
                            try:
                                target_r = int(role_alloc_override.get(rname, 0))
                            except Exception:
                                target_r = 0
                            if target_r <= 0:
                                continue
                            try:
                                curS = float(pd.to_numeric(df_em_roles.at[i, 'Senior'], errors='coerce')) if 'Senior' in cols else 0.0
                                curM = float(pd.to_numeric(df_em_roles.at[i, 'Mid'], errors='coerce')) if 'Mid' in cols else 0.0
                                curJ = float(pd.to_numeric(df_em_roles.at[i, 'Junior'], errors='coerce')) if 'Junior' in cols else 0.0
                            except Exception:
                                curS, curM, curJ = 0.0, 0.0, 0.0
                            total_now_r = float(curS + curM + curJ)
                            if total_now_r <= 0:
                                base = [1.0, 1.0, 0.0]
                            else:
                                base = [curS, curM, curJ]
                            scaled = [v / (sum(base) or 1.0) * float(target_r) for v in base]
                            try:
                                ints = _round_lrm(scaled, total_target=int(target_r))
                            except Exception:
                                ints = [int(round(x)) for x in scaled]
                            Sr, Md, Jr = ints + [0] * (3 - len(ints))
                            if Jr > Sr:
                                shift = Jr - Sr
                                Jr -= shift; Md += shift
                            if Md < Sr and Jr > 0:
                                take = min(Sr - Md, Jr)
                                Jr -= take; Md += take
                            if 'Senior' in cols: df_em.at[i, 'Senior'] = int(Sr)
                            if 'Mid' in cols:    df_em.at[i, 'Mid']    = int(Md)
                            if 'Junior' in cols: df_em.at[i, 'Junior'] = int(Jr)
                except Exception:
                    pass
                target = int(desired_team_size) if desired_team_size else None
                if cols and target is not None and not (role_alloc_override if 'role_alloc_override' in locals() else None):
                    current = pd.to_numeric(df_em[cols], errors="coerce").fillna(0.0)
                    total_now = float(current.to_numpy().sum())
                    if int(total_now) != int(target) and total_now > 0:
                        # Proportional rescale across all cells to hit the exact Team Size
                        flat = current.values.flatten().tolist()
                        scaled = [x / total_now * float(target) for x in flat]
                        ints = _round_lrm(scaled, total_target=int(target))
                        # Write back into the DataFrame in original order
                        import numpy as _np
                        arr = _np.array(ints).reshape(current.shape)
                        for j, c in enumerate(cols):
                            df_em[c] = arr[:, j]
                # Append a Total row for visibility
                if cols:
                    total_row = {"Role": "Total", **{c: int(pd.to_numeric(df_em[c], errors='coerce').fillna(0).sum()) for c in cols}}
                    df_disp = pd.concat([df_em, pd.DataFrame([total_row])], ignore_index=True)
                else:
                    df_disp = df_em
                show_df(df_disp)
            except Exception:
                show_df(exp_mix_df)
    st.subheader("Forecast Table")
    if forecast_table:
        with st.expander("Forecast Table"):
            try:
                df_ft = pd.DataFrame(forecast_table)
                required_cols = {"quarter", "role_family", "level", "demand"}
                if not df_ft.empty and required_cols.issubset(df_ft.columns):
                    # Build a rule-adjusted view for Forecast Table (does not mutate backend plan)
                    df_f = df_ft.copy()
                    # Role totals
                    role_sums = df_f.groupby("role_family", as_index=False)["demand"].sum().rename(columns={"role_family":"Role","demand":"count"})
                    desired_total_team = int(desired_team_size) if desired_team_size else None
                    if 'role_alloc_override' in locals() and role_alloc_override:
                        role_totals = {str(k): int(v) for k, v in role_alloc_override.items() if int(v) > 0}
                    elif desired_total_team:
                        weights = {row["Role"]: float(row["count"]) for _, row in role_sums.iterrows()}
                        total_w = sum(weights.values()) or 1.0
                        scaled = {k: (v / total_w) * float(desired_total_team) for k, v in weights.items()}
                        floors = {k: int(float(v)) for k, v in scaled.items()}
                        residual = int(desired_total_team) - sum(floors.values())
                        order = sorted(scaled.items(), key=lambda kv: (float(kv[1]) - int(float(kv[1]))), reverse=True)
                        for i in range(max(0, residual)):
                            floors[order[i][0]] += 1
                        role_totals = {k: floors[k] for k in floors.keys()}
                    else:
                        role_totals = {row["Role"]: int(round(float(row["count"]))) for _, row in role_sums.iterrows()}
                    # Level buckets and pod ratio
                    buck = {"Lead": "Senior", "Principal": "Senior", "Senior": "Senior", "Mid": "Mid", "Junior": "Junior"}
                    df_f["exp_bucket"] = df_f["level"].map(lambda x: buck.get(str(x), "Mid"))
                    mix = df_f.groupby(["role_family", "exp_bucket"], as_index=False)["demand"].sum()
                    pivot = mix.pivot(index="role_family", columns="exp_bucket", values="demand").fillna(0.0)
                    for col in ["Senior", "Mid", "Junior"]:
                        if col not in pivot.columns:
                            pivot[col] = 0.0
                    pivot = pivot[["Senior", "Mid", "Junior"]]
                    def round_lrm(vals, total_target=None):
                        vals = [float(v) for v in vals]
                        if total_target is None:
                            total_target = int(round(sum(vals)))
                        floors = [int(math.floor(v)) for v in vals]
                        residual = int(max(0, total_target - sum(floors)))
                        if residual > 0 and len(vals) > 0:
                            rem = [v - f for v, f in zip(vals, floors)]
                            order = sorted(range(len(rem)), key=lambda i: rem[i], reverse=True)
                            i = 0
                            while residual > 0 and order:
                                idx = order[i % len(order)]
                                floors[idx] += 1
                                residual -= 1
                                i += 1
                        return floors
                    target_by_rl = {}
                    for role_name, row in pivot.iterrows():
                        target = int(role_totals.get(str(role_name), int(round(row.sum()))))
                        scaled = [float(row.get("Senior", 0.0)), float(row.get("Mid", 0.0)), float(row.get("Junior", 0.0))]
                        total_v = sum(scaled) or 1.0
                        scaled = [v / total_v * target for v in scaled]
                        Sr, Md, Jr = round_lrm(scaled, total_target=target)
                        if Jr > Sr:
                            shift = Jr - Sr
                            Jr -= shift
                            Md += shift
                        if Md < Sr and Jr > 0:
                            take = min(Sr - Md, Jr)
                            Jr -= take
                            Md += take
                        target_by_rl[(str(role_name), "Senior")] = Sr
                        target_by_rl[(str(role_name), "Mid")] = Md
                        target_by_rl[(str(role_name), "Junior")] = Jr
                    g_q = df_f.groupby(["quarter", "role_family", "exp_bucket"], as_index=False).agg({
                        "demand": "sum",
                        "internal_fulfilled": "sum",
                    })
                    quarters_all = sorted(df_f["quarter"].unique().tolist())
                    rows_out = []
                    for (role_name, lvl), total_T in target_by_rl.items():
                        sub = g_q[(g_q["role_family"] == role_name) & (g_q["exp_bucket"] == lvl)].copy()
                        if total_T == 0:
                            continue
                        if sub.empty:
                            sub = pd.DataFrame({
                                "quarter": quarters_all,
                                "role_family": [role_name] * len(quarters_all),
                                "exp_bucket": [lvl] * len(quarters_all),
                                "demand": [1.0] * len(quarters_all),
                                "internal_fulfilled": [0.0] * len(quarters_all),
                            })
                        w = [float(x) for x in sub["demand"].tolist()]
                        if sum(w) <= 0:
                            w = [1.0] * len(w)
                        d_counts = round_lrm([x / sum(w) * total_T for x in w], total_target=total_T)
                        # Leadership first
                        if role_name in ("PM", "Architect") and len(d_counts) > 0:
                            d_total = sum(d_counts)
                            d_counts = [d_total] + [0] * (len(d_counts) - 1)
                        # Senior early
                        eng_roles = ["Backend","Frontend","FullStack","DataEng","DataSci","MLE","Mobile"]
                        if role_name in eng_roles and lvl == "Senior" and len(d_counts) >= 2:
                            early = d_counts[0] + d_counts[1]
                            target_early = int(math.ceil(0.60 * total_T))
                            if early < target_early:
                                short = target_early - early
                                for i in range(2, len(d_counts)):
                                    if short <= 0:
                                        break
                                    take = min(short, d_counts[i])
                                    d_counts[i] -= take
                                    give_q = 0 if d_counts[0] <= d_counts[1] else 1
                                    d_counts[give_q] += take
                                    short -= take
                        # QA shift-left
                        if role_name == "QA" and total_T > 0 and len(d_counts) > 0 and d_counts[0] == 0:
                            for j in range(len(d_counts) - 1, -1, -1):
                                if d_counts[j] > 0:
                                    d_counts[j] -= 1
                                    d_counts[0] += 1
                                    break
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
                                "demand": dc,
                                "internal_fulfilled": ic,
                                "external_hires": ec,
                            })
                    df_view = pd.DataFrame(rows_out)
                    # Attach salary and estimate cost per quarter
                    try:
                        sal = df_ft.groupby(["quarter","role_family","level"], as_index=False)["avg_salary"].mean()
                        df_view = df_view.merge(sal, on=["quarter","role_family","level"], how="left")
                        df_view["cost_estimated"] = (pd.to_numeric(df_view.get("avg_salary"), errors="coerce").fillna(0.0) / 4.0) * df_view["demand"]
                    except Exception:
                        df_view["avg_salary"] = None
                        df_view["cost_estimated"] = 0.0
                    df_view = df_view.sort_values(["quarter","role_family","level"]).reset_index(drop=True)
                    show_df(df_view)
                    forecast_display_df = df_view.copy()
                else:
                    if "technology" in df_ft.columns:
                        df_ft = df_ft.drop(columns=["technology"])  # hide Technology column in display
                    forecast_display_df = df_ft.copy()
                    show_df(df_ft)
            except Exception:
                show_df(forecast_table)
    else:
        st.warning("No forecast rows generated. Try adjusting quarters, region, or JD technologies. If you selected future quarters, the app now regenerates market trends up to the requested end year.")
    forecast_source = (
        forecast_display_df
        if forecast_display_df is not None
        else (pd.DataFrame(forecast_table) if forecast_table else pd.DataFrame())
    )
    st.subheader("Hiring Plan (External)")
    # External hires aggregated by quarter+role so each role appears once per quarter (omit quarters with zero demand)
    hiring_plan_view = None
    try:
        required_cols = {"quarter", "role_family", "external_hires"}
        if not forecast_source.empty and required_cols.issubset(forecast_source.columns):
            df_hp = (
                forecast_source.groupby(["quarter", "role_family"], as_index=False)
                .agg({"external_hires": "sum"})
            )
            df_hp = df_hp[df_hp["external_hires"].astype(float) > 0]
            df_hp = df_hp.sort_values(["quarter", "role_family"])
            hiring_plan_view = df_hp.to_dict(orient='records')
    except Exception:
        hiring_plan_view = None
    hp_rows = hiring_plan_view or hiring_plan or []
    if hp_rows:
        with st.expander("Hiring Plan (External)"):
            show_df(hp_rows)
            try:
                hiring_plan_df = pd.DataFrame(hp_rows)
            except Exception:
                hiring_plan_df = None
    else:
        st.info("No external hires planned (may be due to empty forecast or budget/availability constraints).")
    role_quarter_needs = set()
    if not forecast_source.empty and {"quarter", "role_family", "demand"}.issubset(forecast_source.columns):
        agg_kwargs = {"demand": "sum"}
        if "internal_fulfilled" in forecast_source.columns:
            agg_kwargs["internal_fulfilled"] = "sum"
        df_need = (
            forecast_source.groupby(["quarter", "role_family"], as_index=False)
            .agg(agg_kwargs)
        )
        for _, row in df_need.iterrows():
            demand = float(row["demand"])
            internal = float(row.get("internal_fulfilled", 0.0))
            if demand > 0 and demand > internal:
                role_quarter_needs.add((row["quarter"], row["role_family"]))

    st.subheader("Reassignment Plan (Internal)")
    if reassignment_plan:
        with st.expander("Reassignment Plan (Internal)"):
            try:
                _rp = reassignment_plan
                if 'role_alloc_override' in locals() and role_alloc_override:
                    try:
                        _df_rp = pd.DataFrame(reassignment_plan)
                        _roles = [str(k) for k in (role_alloc_override or {}).keys()]
                        if 'role_family' in _df_rp.columns:
                            _df_rp = _df_rp[_df_rp['role_family'].astype(str).isin(_roles)]
                        _rp = _df_rp.to_dict(orient='records')
                    except Exception:
                        pass
                if _rp:
                    try:
                        df_agg = pd.DataFrame(_rp)
                        group_cols = ["employee_id", "quarter", "role_family", "level"]
                        agg_kwargs = {"fte": "sum"}
                        if "technology" in df_agg.columns:
                            agg_kwargs["technology"] = lambda vals: ", ".join(sorted({str(v) for v in vals if v}))
                        df_agg = df_agg.groupby(group_cols, as_index=False).agg(agg_kwargs)
                        if role_quarter_needs:
                            df_agg = df_agg[
                                df_agg.apply(
                                    lambda row: (row["quarter"], row["role_family"]) in role_quarter_needs,
                                    axis=1,
                                )
                            ]
                        _rp = df_agg.to_dict(orient="records")
                    except Exception:
                        pass
                show_df(_rp)
            except Exception:
                show_df(reassignment_plan)
            try:
                reassignment_plan_df = pd.DataFrame(reassignment_plan)
            except Exception:
                reassignment_plan_df = None
    else:
        st.info("No reassignments planned (may be due to empty forecast or no internal releases in chosen quarters).")
    st.subheader("Upcoming Internal Releases (+/- 1 month of start)")
    if release_suggestions:
        with st.expander("Upcoming Internal Releases (+/- 1 month of start)"):
            try:
                _rel = release_suggestions
                if 'role_alloc_override' in locals() and role_alloc_override:
                    try:
                        _df_rel = pd.DataFrame(release_suggestions)
                        _roles = [str(k) for k in (role_alloc_override or {}).keys()]
                        if 'role' in _df_rel.columns:
                            _df_rel = _df_rel[_df_rel['role'].astype(str).isin(_roles)]
                        _rel = _df_rel.to_dict(orient='records')
                    except Exception:
                        pass
                show_df(_rel)
            except Exception:
                show_df(release_suggestions)
    else:
        st.info("No matching internal releases near the start quarter for requested roles.")
    # Totals Summary (moved just above Charts)
    df_tot = pd.DataFrame(forecast_table) if forecast_table else pd.DataFrame()
    quarters = sorted(set(df_tot["quarter"])) if not df_tot.empty and "quarter" in df_tot.columns else []
    n_quarters = len(quarters)
    per_quarter_target = active_team_size if active_team_size is not None else (int(desired_team_size) if desired_team_size else None)
    desired_team_total = (int(per_quarter_target) * n_quarters) if per_quarter_target else None
    forecast_total = float(df_tot["demand"].sum()) if not df_tot.empty and "demand" in df_tot.columns else 0.0
    internal_total = float(df_tot["internal_fulfilled"].sum()) if not df_tot.empty and "internal_fulfilled" in df_tot.columns else 0.0
    external_total = float(df_tot["external_hires"].sum()) if not df_tot.empty and "external_hires" in df_tot.columns else 0.0
    total_cost = float(df_tot["cost_estimated"].sum()) if not df_tot.empty and "cost_estimated" in df_tot.columns else 0.0
    profit_val = (float(budget) - total_cost) if (budget is not None) else None
    # Profit guardrail (effective budget after margin)
    eff_budget_display = float(budget) * (1.0 - float(min_profit_margin or 0.0)/100.0) if budget is not None else None
    profit_target_amt = float(budget) * float(min_profit_margin or 0.0)/100.0 if budget is not None else None
    meets_margin = (profit_val is not None and profit_target_amt is not None and profit_val >= profit_target_amt)
    quarterly_required_cols = {"quarter", "demand", "internal_fulfilled", "external_hires", "cost_estimated"}
    df_quarterly = pd.DataFrame()
    if not df_tot.empty and quarterly_required_cols.issubset(set(df_tot.columns)):
        df_quarterly = (
            df_tot.groupby("quarter", as_index=False)
            .agg({
                "demand": "sum",
                "internal_fulfilled": "sum",
                "external_hires": "sum",
                "cost_estimated": "sum",
            })
        )
        for numeric_col in ["demand", "internal_fulfilled", "external_hires", "cost_estimated"]:
            df_quarterly[numeric_col] = pd.to_numeric(df_quarterly[numeric_col], errors="coerce").fillna(0.0)
        df_quarterly["total_supply"] = (
            df_quarterly["internal_fulfilled"] + df_quarterly["external_hires"]
        )
    df_profit_quarters = pd.DataFrame()
    quarters_meeting_margin = 0
    quarters_missing_margin = 0
    avg_margin_pct = None
    if not df_quarterly.empty:
        df_profit_quarters = df_quarterly.copy()
        df_profit_quarters["cost_estimated"] = df_quarterly["cost_estimated"].copy()
        if budget is not None:
            budget_value = float(budget)
            if total_cost > 0:
                df_profit_quarters["budget_allocated"] = (
                    df_profit_quarters["cost_estimated"] / total_cost
                ) * budget_value
            else:
                per_quarter_budget = budget_value / max(1, len(df_profit_quarters))
                df_profit_quarters["budget_allocated"] = per_quarter_budget
        else:
            df_profit_quarters["budget_allocated"] = np.nan
        df_profit_quarters["profit"] = (
            df_profit_quarters["budget_allocated"] - df_profit_quarters["cost_estimated"]
        )
        df_profit_quarters["margin_pct"] = np.where(
            df_profit_quarters["budget_allocated"].notna() & (df_profit_quarters["budget_allocated"] != 0),
            df_profit_quarters["profit"] / df_profit_quarters["budget_allocated"] * 100,
            np.nan,
        )
        target_margin = float(min_profit_margin or 0.0)
        df_profit_quarters["meets_margin"] = (
            (df_profit_quarters["margin_pct"] >= target_margin).fillna(False)
        )
        valid_margin = df_profit_quarters["margin_pct"].dropna()
        avg_margin_pct = round(float(valid_margin.mean()), 2) if not valid_margin.empty else None
        quarters_meeting_margin = int(df_profit_quarters["meets_margin"].sum())
        quarters_missing_margin = int(max(0, len(df_profit_quarters) - quarters_meeting_margin))
    summary_rows = [{
        "quarters": n_quarters,
        "desired_team_size": int(per_quarter_target) if per_quarter_target is not None else None,
        "forecast_total_demand": round(forecast_total, 2),
        "internal_total": round(internal_total, 2),
        "profit": round(profit_val, 2) if profit_val is not None else None,
        "profitable": (profit_val is not None and profit_val >= 0),
        "profit_margin_pct": int(min_profit_margin) if min_profit_margin is not None else 0,
        "profit_target": round(profit_target_amt, 2) if profit_target_amt is not None else None,
        "effective_budget": round(eff_budget_display, 2) if eff_budget_display is not None else None,
        "meets_margin": bool(meets_margin),
        "avg_profit_margin_pct": avg_margin_pct,
        "quarters_meeting_margin": quarters_meeting_margin,
        "quarters_missing_margin": quarters_missing_margin,
    }]
    with st.expander("Totals Summary"):
        show_df(summary_rows)
        try:
            totals_summary_df = pd.DataFrame(summary_rows)
        except Exception:
            totals_summary_df = None
        if not df_profit_quarters.empty:
            st.markdown("#### Profit Insights by Quarter")
            profit_display_cols = ["quarter", "budget_allocated", "cost_estimated", "profit", "margin_pct", "meets_margin"]
            df_profit_display = df_profit_quarters[profit_display_cols].copy()
            df_profit_display["budget_allocated"] = df_profit_display["budget_allocated"].round(2)
            df_profit_display["profit"] = df_profit_display["profit"].round(2)
            df_profit_display["margin_pct"] = df_profit_display["margin_pct"].round(2)
            show_df(df_profit_display)
    if profit_val is not None and profit_val < 0:
        st.error("Profit is negative. Consider reducing team size, shifting to internal reassignments, or increasing budget.")
    elif profit_val is not None and not meets_margin:
        quarters_for_margin = len(df_profit_quarters) if not df_profit_quarters.empty else 0
        margin_hint = (
            f" {quarters_missing_margin}/{quarters_for_margin} quarter(s) fall below the target margin."
            if quarters_for_margin
            else ""
        )
        st.warning(f"Plan meets budget but misses the profit margin target.{margin_hint}")
    # Charts (wrapped in a single expander and controlled by toggle)
    if show_charts:
        if px:
            with st.expander("Charts"):
                df = pd.DataFrame(forecast_table)
                required_cols = {"quarter", "role_family", "demand", "internal_fulfilled", "external_hires", "cost_estimated"}
                if not df.empty and required_cols.issubset(set(df.columns)):
                    if not df_quarterly.empty:
                        df_sum = df_quarterly.copy()
                    else:
                        df_sum = df.groupby("quarter").agg({
                            "demand": "sum",
                            "internal_fulfilled": "sum",
                            "external_hires": "sum",
                            "cost_estimated": "sum",
                        }).reset_index()
                        for col in ["internal_fulfilled", "external_hires"]:
                            df_sum[col] = pd.to_numeric(df_sum[col], errors="coerce").fillna(0.0)
                        df_sum["total_supply"] = (
                            df_sum["internal_fulfilled"] + df_sum["external_hires"]
                        )
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### Quarterly Resource Demand by Role")
                        mix = df.groupby(["quarter", "role_family"]).agg({"demand": "sum"}).reset_index()
                        fig_roles = px.line(
                            mix,
                            x="quarter",
                            y="demand",
                            color="role_family",
                            markers=True,
                            title="",
                        )
                        fig_roles.update_layout(yaxis_title="Required Count", legend_title="")
                        st.plotly_chart(fig_roles, use_container_width=True)
                    with col2:
                        st.markdown("#### Cost Projection by Quarter")
                        fig_cost = px.line(
                            df_sum,
                            x="quarter",
                            y="cost_estimated",
                            markers=True,
                            title="",
                        )
                        fig_cost.update_layout(yaxis_title="Cost (USD)")
                        st.plotly_chart(fig_cost, use_container_width=True)
                    col3, col4 = st.columns(2)
                    with col3:
                        st.markdown("#### Internal vs External Supply by Quarter")
                        fig_supply = px.bar(
                            df_sum,
                            x="quarter",
                            y=["internal_fulfilled", "external_hires"],
                            title="",
                            barmode="stack",
                        )
                        fig_supply.update_layout(yaxis_title="Count", legend_title="")
                        if "total_supply" in df_sum.columns:
                            max_total_supply = float(df_sum["total_supply"].max())
                            if max_total_supply > 0:
                                fig_supply.update_layout(
                                    yaxis=dict(range=[0, max_total_supply * 1.1])
                                )
                            fig_supply.update_traces(texttemplate="%{y:.1f}", textposition="inside")
                        st.plotly_chart(fig_supply, use_container_width=True)
                    with col4:
                        if budget:
                            if go:
                                st.markdown("#### Budget vs Projected Cost")
                                total_cost = df_sum["cost_estimated"].sum()
                                percentage = round(total_cost / budget * 100, 2) if budget else 0.0
                                fig_gauge = go.Figure(go.Indicator(
                                    mode="gauge+number",
                                    value=percentage,
                                    gauge={
                                        "shape": "angular",
                                        "axis": {"range": [0, 150]},
                                        "bar": {"color": "#1f77b4"},
                                        "steps": [
                                            {"range": [0, 80], "color": "#90ee90"},
                                            {"range": [80, 100], "color": "#fdd835"},
                                            {"range": [100, 120], "color": "#fb8c00"},
                                            {"range": [120, 150], "color": "#e53935"},
                                        ],
                                    },
                                    title={"text": "Budget % Used"},
                                ))
                                fig_gauge.update_layout(margin=dict(t=20, b=20, l=20, r=20))
                                st.plotly_chart(fig_gauge, use_container_width=True)
                            else:
                                st.info("Install `plotly.graph_objects` to view the budget gauge.")
                        else:
                            st.info("Provide a budget to visualize budget vs projected cost.")

                    role_dist = (
                        df.groupby("role_family", as_index=False)
                        .agg({"demand": "sum"})
                        .rename(columns={"demand": "required_count"})
                    )
                    if not role_dist.empty:
                        st.markdown("#### Role Distribution")
                        fig_role_dist = px.pie(
                            role_dist,
                            names="role_family",
                            values="required_count",
                            title="",
                            hole=0,
                        )
                        fig_role_dist.update_traces(
                            textposition="inside",
                            textinfo="percent",
                            hovertemplate="role=%{label}<br>required_count=%{value}",
                        )
                        fig_role_dist.update_layout(
                            margin=dict(t=20, b=20, l=20, r=20),
                            legend_title="",
                        )
                        st.plotly_chart(fig_role_dist, use_container_width=True)
                else:
                    st.info("No forecast data available to plot yet.")
        else:
            st.info("Install plotly for charts: pip install plotly")


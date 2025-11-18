import csv
import pandas as pd
import math
import os
import random
from datetime import datetime, timedelta

from .utils.ontology import (
    DOMAINS,
    REGIONS,
    ROLE_FAMILIES,
    LEVELS,
    TECHNOLOGIES,
    TECH_GROUP,
    ROLE_TECH_ALIGNMENT,
    region_cost_multiplier,
)


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def quarter_iter(start_year=2000, end_year=2025):
    for y in range(start_year, end_year + 1):
        for q in range(1, 5):
            yield y, q


def seasonal_boost(q: int, domain: str) -> float:
    base = {1: 1.05, 2: 0.98, 3: 1.0, 4: 1.06}[q]
    if domain in ("FinTech", "E-commerce"):
        # More volatile
        base *= {1: 1.08, 2: 0.95, 3: 0.98, 4: 1.1}[q]
    if domain in ("Healthcare", "Public Sector"):
        base *= {1: 1.0, 2: 1.0, 3: 1.01, 4: 1.0}[q]
    return base


def long_run_trend(year: int, tech: str) -> float:
    group = TECH_GROUP.get(tech, "App")
    t = year
    if group == "Cloud":
        # Accelerates post-2014
        base = 20 + 2.0 * max(0, t - 2010) + 1.5 * max(0, t - 2014)
    elif group == "Data":
        # Rising strongly post-2015; extra pop post-2020
        base = 15 + 2.0 * max(0, t - 2012) + 2.0 * max(0, t - 2015) + 3.0 * max(0, t - 2020)
    elif group == "QA":
        base = 25 + 1.0 * max(0, t - 2008)
    else:
        # App dev steady rise
        base = 30 + 1.2 * max(0, t - 2005)
    return base


def macro_shocks(year: int) -> float:
    shock = 1.0
    if year in (2008, 2009):
        shock *= 0.9
    if year in (2020,):
        shock *= 0.85
    if year in (2022, 2023):
        shock *= 0.95
    return shock


def gen_market_trends(end_year: int | None = None):
    path = os.path.join(DATA_DIR, "market_trends_quarterly.csv")
    parquet_path = os.path.join(DATA_DIR, "market_trends_quarterly.parquet")
    random.seed(42)
    rows = []
    end_y = end_year or 2030
    for year, q in quarter_iter(2000, end_y):
        for domain in DOMAINS:
            for tech in TECHNOLOGIES:
                # Generate a few region-specific rows; some unknown region
                for region in (REGIONS + [""]):
                    # Demand trend
                    base = long_run_trend(year, tech)
                    seas = seasonal_boost(q, domain)
                    shock = macro_shocks(year)
                    noise = random.uniform(-2, 2)
                    demand = max(0.0, min(100.0, base * seas * shock / 2.0 + noise))

                    # Remote ratio: low before 2020, jumps after
                    if year < 2018:
                        remote_ratio = random.uniform(0.05, 0.15)
                    elif year < 2020:
                        remote_ratio = random.uniform(0.1, 0.25)
                    elif year < 2022:
                        remote_ratio = random.uniform(0.4, 0.7)
                    else:
                        remote_ratio = random.uniform(0.35, 0.65)

                    # Salary driven by demand + inflation
                    # Region multiplier and tech premium applied on annual basis
                    # Earlier version divided by 30 and collapsed values to the floor (20000). Fix by removing that divisor
                    inflation = 1.0 + 0.02 * max(0, year - 2000)
                    region_mult = region_cost_multiplier(region if region else None)
                    tech_premium = 1.15 if TECH_GROUP.get(tech, "App") in ("Cloud", "Data") else 1.0
                    # Baseline: 30k rising with demand; scale by inflation, region, tech premium
                    median_salary = max(25000.0, (30000 + demand * 1200) * inflation * region_mult * tech_premium)

                    # Hiring velocity inversely with demand
                    hiring_velocity = max(0.05, min(1.0, 1.0 - demand / 120.0 + random.uniform(-0.05, 0.05)))

                    # Attrition slightly higher with hot markets
                    attrition = max(0.02, min(0.25, 0.08 + demand / 600.0 + random.uniform(-0.02, 0.02)))

                    rows.append([
                        year, q, domain, tech, region,
                        round(demand, 2), round(median_salary, 2), round(hiring_velocity, 3), round(attrition, 3),
                        round(remote_ratio, 3)
                    ])

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "year", "quarter", "domain", "technology", "region",
            "demand_index", "median_salary", "hiring_velocity", "attrition_rate_est", "remote_ratio"
        ])
        w.writerows(rows)
    # Also persist Parquet for faster internal loads (optional if pyarrow/fastparquet available)
    try:
        df = pd.DataFrame(rows, columns=[
            "year", "quarter", "domain", "technology", "region",
            "demand_index", "median_salary", "hiring_velocity", "attrition_rate_est", "remote_ratio"
        ])
        df.to_parquet(parquet_path, index=False)
    except Exception:
        # Parquet writer not available; CSV remains as canonical output
        pass
    return path


def random_name():
    first = ["Asha", "Ravi", "Priya", "Neha", "Kunal", "Meera", "Ankit", "Ishan", "Sara", "Vikram",
             "Emily", "James", "Liu", "Wei", "Anna", "Carlos", "Fatima", "Omar", "Sofia", "Elena"]
    last = ["Patel", "Sharma", "Iyer", "Khan", "Mehta", "Joshi", "Kumar", "Singh", "Desai", "Naidu",
            "Smith", "Johnson", "Garcia", "Miller", "Lopez", "Williams", "Chen", "Zhang", "Silva", "Ivanov"]
    return random.choice(first) + " " + random.choice(last)


def gen_employees_projects_assignments():
    emp_path = os.path.join(DATA_DIR, "employees.csv")
    emp_parquet = os.path.join(DATA_DIR, "employees.parquet")
    proj_path = os.path.join(DATA_DIR, "projects.csv")
    proj_parquet = os.path.join(DATA_DIR, "projects.parquet")
    asg_path = os.path.join(DATA_DIR, "assignments.csv")
    asg_parquet = os.path.join(DATA_DIR, "assignments.parquet")
    random.seed(7)

    # Employees
    employees = []
    employee_id = 1000
    # Size based on projects; we will create 30-120 projects and size team accordingly
    num_employees = random.randint(180, 400)
    for _ in range(num_employees):
        name = random_name()
        role = random.choice(ROLE_FAMILIES)
        # Avoid empty tech roles like UI/UX -> give general tags
        aligned = ROLE_TECH_ALIGNMENT.get(role, [])
        primary = random.choice(aligned) if aligned else random.choice(TECHNOLOGIES)
        secondary_list = list(set(random.sample(TECHNOLOGIES, k=random.randint(1, 4)) + aligned[:2]))
        level = random.choices(LEVELS, weights=[20, 35, 30, 12, 3])[0]
        exp_years = max(0.5, random.gauss(6, 3))
        base_location = random.choice(["Bengaluru", "Hyderabad", "Pune", "Delhi NCR", "Mumbai", "Chennai", "Austin", "Toronto", "Berlin", "Singapore"])
        # Map location to region roughly
        region = "India" if base_location in ["Bengaluru", "Hyderabad", "Pune", "Delhi NCR", "Mumbai", "Chennai"] else (
            "NA" if base_location in ["Austin", "Toronto"] else ("EU" if base_location == "Berlin" else "APAC")
        )
        cost_mult = region_cost_multiplier(region)
        level_mult = {"Junior": 0.7, "Mid": 1.0, "Senior": 1.4, "Lead": 1.7, "Principal": 2.1}[level]
        tech_premium = 1.1 if TECH_GROUP.get(primary, "App") in ("Cloud", "Data") else 1.0
        ctc = 20000 * cost_mult * level_mult * tech_premium + random.uniform(-2000, 2000)
        employees.append({
            "employee_id": employee_id,
            "name": name,
            "role_family": role,
            "level": level,
            "experience_years": round(exp_years, 1),
            "primary_tech": primary,
            "secondary_techs": ",".join(secondary_list[:5]),
            "base_location": base_location,
            "cost_to_company_per_year": round(ctc, 2),
        })
        employee_id += 1

    # Projects
    projects = []
    assignments = []
    project_id = 5000
    today = datetime(2025, 4, 1)
    num_projects = random.randint(30, 120)
    for _ in range(num_projects):
        domain = random.choice(DOMAINS)
        region = random.choice(REGIONS + [""])
        project_name = f"{domain} Solution {project_id}"
        start_offset_days = random.randint(-365, 365)
        duration_days = random.randint(90, 540)
        start_date = today + timedelta(days=start_offset_days)
        end_date_planned = start_date + timedelta(days=duration_days)
        contract_type = random.choice(["T&M", "Fixed", "Retainer"])

        # Team shape rules
        team = []
        team.append(("PM", "Senior", 1))
        lead_role = random.choice(["Architect", "LeadAnchor"])  # internal marker
        if lead_role == "Architect":
            team.append(("Architect", random.choice(["Senior", "Lead", "Principal"]), 1))
        # Engineers
        ic_count = random.randint(3, 8)
        for _i in range(ic_count):
            role = random.choice(["Backend", "Frontend", "FullStack", "DataEng", "DevOps", "DataSci"])
            level = random.choices(LEVELS, weights=[20, 35, 30, 12, 3])[0]
            team.append((role, level, 1))
        # QA
        for _i in range(random.randint(1, 3)):
            team.append(("QA", random.choice(["Mid", "Senior"]), 1))
        # Optional DevOps/Data extra
        for _i in range(random.randint(0, 2)):
            role = random.choice(["DevOps", "DataEng"])
            team.append((role, random.choice(["Mid", "Senior", "Lead"]), 1))

        # Choose a rough tech stack consistent with domain
        domain_pref = {
            "FinTech": ["Java", "React", "Kafka", "SQL", "AWS"],
            "E-commerce": ["Node.js", "React", "AWS", "Snowflake"],
            "Healthcare": ["Python", "Angular", "Azure", "Airflow", "Tableau/PowerBI"],
            "Telecom": ["Java", "Kubernetes", "Kafka", "GCP"],
            "Manufacturing": [".NET", "Angular", "Azure", "SQL"],
            "Public Sector": ["Java", "React", "AWS", "SQL"],
            "Insurance": ["Java", "Spark", "Kafka", "Snowflake"],
            "Energy": ["Python", "Databricks", "Azure"],
            "Retail": ["Node.js", "React", "AWS", "Snowflake"],
        }
        preferred = domain_pref.get(domain, ["Java", "React", "AWS"])[:]
        tech_stack = list(set(preferred + random.sample(TECHNOLOGIES, k=random.randint(1, 3))))

        # Budget: sum of estimated monthly bill for planned team
        region_mult = region_cost_multiplier(region if region else None)
        base_month_rate = 8000 * region_mult
        level_mult_map = {"Junior": 0.8, "Mid": 1.0, "Senior": 1.35, "Lead": 1.6, "Principal": 2.0}
        planned_months = max(3, int(duration_days / 30))
        team_month_cost = 0.0
        for role, level, count in team:
            team_month_cost += base_month_rate * level_mult_map.get(level, 1.0) * count
        # Add margin/random variance
        budget_total = team_month_cost * planned_months * random.uniform(0.9, 1.2)

        projects.append({
            "project_id": project_id,
            "project_name": project_name,
            "client_domain": domain,
            "region": region,
            "start_date": start_date.date().isoformat(),
            "end_date_planned": end_date_planned.date().isoformat(),
            "budget_total": round(budget_total, 2),
            "contract_type": contract_type,
        })

        # Assignments: map employees with skill alignment and availability
        # Approx allocation 1.0 each, some at 0.5; backfills occasional
        needed_roles = [r for r, _lvl, _c in team for _ in range(1)]
        pool = employees[:]  # naive; could filter by region to be tighter
        random.shuffle(pool)
        assigned = 0
        for (role, level, count) in team:
            for _ in range(count):
                matches = [e for e in pool if e["role_family"] == role]
                if not matches:
                    matches = pool[:]
                e = random.choice(matches)
                pool.remove(e)
                assigned += 1
                allocation = random.choice([1.0, 1.0, 1.0, 0.5])
                bill_rate = base_month_rate * {"Junior": 0.9, "Mid": 1.0, "Senior": 1.4, "Lead": 1.7, "Principal": 2.1}.get(level, 1.0)
                is_backfilled = random.random() < 0.12
                notice_days = random.choice([30, 45, 60, 90])
                asg_start = start_date + timedelta(days=random.randint(-15, 30))
                asg_end = min(end_date_planned + timedelta(days=random.randint(-30, 60)), start_date + timedelta(days=540))
                assignments.append({
                    "assignment_id": len(assignments) + 1,
                    "project_id": project_id,
                    "employee_id": e["employee_id"],
                    "role_on_project": f"{role}-{level}",
                    "assigned_start_date": asg_start.date().isoformat(),
                    "assigned_end_date_planned": asg_end.date().isoformat(),
                    "allocation_pct": allocation,
                    "tech_stack": ",".join(tech_stack[:8]),
                    "bill_rate_per_month": round(bill_rate, 2),
                    "is_backfilled": str(is_backfilled).lower(),
                    "notice_period_days": notice_days,
                })

        project_id += 1

    # Filter to ongoing projects that started before 2025-10-01
    filter_cutoff = datetime(2025, 10, 1).date()
    keep_projects = []
    keep_ids = set()
    for p in projects:
        try:
            sd = datetime.fromisoformat(p["start_date"]).date()
            ed = datetime.fromisoformat(p["end_date_planned"]).date()
        except Exception:
            continue
        # Include projects that started before cutoff and are ongoing at or after cutoff
        if sd < filter_cutoff and ed >= filter_cutoff:
            keep_projects.append(p)
            keep_ids.add(p["project_id"])

    # If filtering removed everything (unlikely), fall back to original list
    if keep_projects:
        projects = keep_projects

        # Keep only assignments for kept projects and renumber assignment_id
        new_assignments = []
        aid = 1
        for a in assignments:
            if a.get("project_id") in keep_ids:
                a = dict(a)
                a["assignment_id"] = aid
                new_assignments.append(a)
                aid += 1
        assignments = new_assignments

    # Write CSVs
    with open(emp_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "employee_id", "name", "role_family", "level", "experience_years", "primary_tech",
            "secondary_techs", "base_location", "cost_to_company_per_year"
        ])
        w.writeheader(); w.writerows(employees)

    with open(proj_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "project_id", "project_name", "client_domain", "region", "start_date",
            "end_date_planned", "budget_total", "contract_type"
        ])
        w.writeheader(); w.writerows(projects)

    with open(asg_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "assignment_id", "project_id", "employee_id", "role_on_project", "assigned_start_date",
            "assigned_end_date_planned", "allocation_pct", "tech_stack", "bill_rate_per_month",
            "is_backfilled", "notice_period_days"
        ])
        w.writeheader(); w.writerows(assignments)
    # Also persist Parquet copies for faster internal reads
    try:
        pd.DataFrame(employees).to_parquet(emp_parquet, index=False)
        pd.DataFrame(projects).to_parquet(proj_parquet, index=False)
        pd.DataFrame(assignments).to_parquet(asg_parquet, index=False)
    except Exception:
        pass

    return emp_path, proj_path, asg_path


if __name__ == "__main__":
    p1 = gen_market_trends()
    p2, p3, p4 = gen_employees_projects_assignments()
    # Paths returned for interactive use; suppress noisy prints in library contexts

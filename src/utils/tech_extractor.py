import re
from typing import Dict, List, Tuple

from .ontology import ROLE_FAMILIES, TECHNOLOGIES, TECH_GROUP


def _priority_for(tech: str) -> int:
    return 3 if TECH_GROUP.get(tech, "App") in ("Cloud", "Data") else 2


def _canonical_tech(token: str) -> str | None:
    t = token.strip().lower()
    # Aliases/synonyms
    aliases = {
        "asp.net": "ASP.NET",
        "aspnet": "ASP.NET",
        "ado.net": "Ado.net",
        "ado net": "Ado.net",
        "c#": "C#",
        "csharp": "C#",
        ".net core": ".NET",
        ".net": ".NET",
        "angular cli": "Angular",
        "oracle database": "OracleDB",
        "oracle db": "OracleDB",
        "postgres": "PostgreSQL",
        "ms sql": "SQL Server",
        "sqlserver": "SQL Server",
        "azure devops": "DevOps",
        "github actions": "DevOps",
        "jenkins": "DevOps",
        "ci/cd": "DevOps",
        "cicd": "DevOps",
        "azure service bus": "Azure",
        "azure services": "Azure",
        "key vault": "Azure",
        "app service": "Azure",
        "container registry": "Azure",
        "app insights": "Azure",
        "rest api": "ASP.NET",
        "rest apis": "ASP.NET",
        "web api": "ASP.NET",
        "swagger": "DevOps",
        "postman": "DevOps",
        # Added canonicalizations
        "pl/sql": "PL/SQL",
        "plsql": "PL/SQL",
        "control m": "Control-M",
        "control-m": "Control-M",
        "informatica intelligent cloud services": "IICS",
        "informatica cloud": "IICS",
        "iics": "IICS",
        "business objects": "Business Objects",
        "sap bo": "Business Objects",
        "sap businessobjects": "Business Objects",
        "bo": "Business Objects",
        "amazon s3": "S3",
        "aws s3": "S3",
        "ec2": "EC2",
        "lambda": "Lambda",
        "perl": "Perl",
    }
    for k, v in aliases.items():
        if k in t:
            return v
    # Exact (case-insensitive)
    for tech in TECHNOLOGIES:
        if tech.lower() == t:
            return tech
    # Substring match: prefer longest candidate
    cands = [tech for tech in TECHNOLOGIES if tech.lower() in t or t in tech.lower()]
    if cands:
        return sorted(cands, key=lambda x: -len(x))[0]
    return None


def parse_strict_tech_roles(tech_text: str) -> Tuple[List[Dict], Dict[str, int]]:
    """Parse strict format lines per role:
    Example lines:
      Frontend: React 3y, Angular 2y, JavaScript 4 years
      DataEng (3): Spark 3+, Kafka 2+, Airflow 2y
      DevOps x2: Kubernetes 2y, AWS 2y

    Returns (tech_requirements, role_counts)
    """
    tech_requirements: List[Dict] = []
    role_counts: Dict[str, int] = {}
    if not tech_text:
        return tech_requirements, role_counts

    lines = [l.strip() for l in tech_text.splitlines() if l.strip()]
    header_re = re.compile(r"^(?P<role>[A-Za-z/]+)\s*(?:\((?P<count1>\d+)\)|x\s*(?P<count2>\d+))?\s*:\s*(?P<body>.+)$", re.I)
    item_re = re.compile(r"(?P<name>[A-Za-z0-9#\.\+/ ]+?)\s*(?P<yrs>\d{1,2})\s*(?:\+)?\s*(?:y|yr|yrs|years)?\b", re.I)

    non_header_lines: List[str] = []
    for ln in lines:
        m = header_re.match(ln)
        if not m:
            non_header_lines.append(ln)
            continue
        role_raw = m.group("role").strip()
        # Normalize role to ontology value if possible
        role = next((r for r in ROLE_FAMILIES if r.lower() == role_raw.lower()), None)
        if not role:
            # Allow a couple of synonyms
            alias = {
                "frontend": "Frontend",
                "backend": "Backend",
                "fullstack": "FullStack",
                "datasci": "DataSci",
                "dataeng": "DataEng",
                "pm": "PM",
            }.get(role_raw.lower())
            role = alias if alias in ROLE_FAMILIES else None
        if not role:
            continue

        cnt = 1
        if m.group("count1") or m.group("count2"):
            try:
                cnt = int((m.group("count1") or m.group("count2") or "1"))
            except Exception:
                cnt = 1
        role_counts[role] = role_counts.get(role, 0) + max(1, cnt)

        body = m.group("body")
        for im in item_re.finditer(body):
            name = im.group("name").strip()
            yrs = float(im.group("yrs"))
            tech = _canonical_tech(name)
            if not tech:
                continue
            tech_requirements.append({
                "technology": tech,
                "priority": _priority_for(tech),
                "min_exp_years": yrs,
            })

    # If no role headers or additional lines exist, try free-form extraction per line
    if non_header_lines:
        extra_pairs = _extract_freeform_pairs("\n".join(non_header_lines))
        # Merge, preferring explicit entries already captured
        have = {d["technology"] for d in tech_requirements}
        for tech, yrs in extra_pairs.items():
            if tech not in have:
                tech_requirements.append({
                    "technology": tech,
                    "priority": _priority_for(tech),
                    "min_exp_years": yrs,
                })

    return tech_requirements, role_counts


def _extract_freeform_pairs(text: str) -> Dict[str, float]:
    """Extract technology -> years from freeform bullet lines.
    Handles patterns like 'Tech - 3+ Years', '3 years Tech', 'Tech 2.5 yrs'.
    """
    out: Dict[str, float] = {}
    t = text
    # Build sorted tech names (longest first) for greedy matching
    alias_to_canon = {
        "ASP.NET": "ASP.NET",
        "Ado.net": "Ado.net",
        "C#": "C#",
        ".NET": ".NET",
        "Angular": "Angular",
        "Oracle Database": "OracleDB",
        "Oracle DB": "OracleDB",
        "Postgre": "PostgreSQL",
        "Azure DevOps": "DevOps",
        "GitHub Actions": "DevOps",
        "Jenkins": "DevOps",
        "CI/CD": "DevOps",
        "CICD": "DevOps",
        "Azure Service Bus": "Azure",
        "Azure Services": "Azure",
        "Key Vault": "Azure",
        "App Service": "Azure",
        "Container Registry": "Azure",
        "App Insights": "Azure",
        "REST API": "ASP.NET",
        "REST APIs": "ASP.NET",
        "Web API": "ASP.NET",
        "Web APIs": "ASP.NET",
        "Swagger": "DevOps",
        "Postman": "DevOps",
    }
    tech_names = sorted(list(set(TECHNOLOGIES + list(alias_to_canon.keys()))), key=lambda s: -len(s))
    for tech in tech_names:
        patt_tech = re.escape(tech)
        # number after tech (units optional)
        r1 = re.compile(rf"{patt_tech}[^\n\r\.;:,\)]{{0,100}}?(\d+(?:\.\d+)?)\s*\+?\s*(?:y|yr|yrs|years)?\b", re.I)
        # number before tech (units optional)
        r2 = re.compile(rf"(\d+(?:\.\d+)?)\s*\+?\s*(?:y|yr|yrs|years)?[^\n\r\.;:,\(]{{0,100}}?{patt_tech}", re.I)
        yrs: float | None = None
        m1 = r1.search(t)
        m2 = r2.search(t) if not m1 else None
        if m1:
            try:
                yrs = float(m1.group(1))
            except Exception:
                yrs = None
        elif m2:
            try:
                yrs = float(m2.group(1))
            except Exception:
                yrs = None
        if yrs is not None:
            # Map alias to canonical if needed
            can = alias_to_canon.get(tech) or _canonical_tech(tech) or tech
            prev = out.get(can)
            out[can] = max(prev or 0.0, yrs)
    return out

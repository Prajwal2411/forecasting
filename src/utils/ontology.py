import math
import re
from typing import Any, Dict

from ..config_loader import load_role_taxonomy_config, load_hiring_rule_config


DOMAINS = [
    "FinTech", "E-commerce", "Healthcare", "Telecom", "Manufacturing", "Public Sector",
    "Insurance", "Energy", "Retail"
]

REGIONS = ["India", "NA", "EU", "APAC"]

ROLE_FAMILIES = [
    "PM", "Architect", "Backend", "Frontend", "FullStack", "QA", "DevOps",
    "DataEng", "DataSci", "MLE", "Mobile", "UI/UX", "Support"
]

LEVELS = ["Junior", "Mid", "Senior", "Lead", "Principal"]

TECHNOLOGIES = [
    "Python", "Java", ".NET", "Node.js", "React", "Angular", "JavaScript", "TypeScript", "HTML", "CSS", "AJAX",
    "ASP.NET", "C#", "Ado.net", "MVC",
    "SQL", "PL/SQL", "SQL Server", "PostgreSQL", "OracleDB",
    "AWS", "S3", "EC2", "Lambda", "Azure", "GCP",
    "Spark", "Kafka", "Airflow",
    "Docker", "Kubernetes", "Terraform", "Snowflake", "Databricks",
    "Control-M", "IICS", "Business Objects", "Perl",
    "Tableau/PowerBI", "QA-Manual", "QA-Automation", "DevOps", "MLOps",
    "DataEngineer", "DataScientist"
]

# Grouping for trend logic
TECH_GROUP = {
    # App
    "Python": "App", "Java": "App", ".NET": "App", "Node.js": "App",
    "React": "App", "Angular": "App", "JavaScript": "App", "HTML": "App", "CSS": "App", "AJAX": "App",
    "ASP.NET": "App", "C#": "App", "Ado.net": "App", "MVC": "App",
    "SQL": "Data", "PL/SQL": "Data", "SQL Server": "Data", "PostgreSQL": "Data", "OracleDB": "Data",
    # Cloud/Data/Platform
    "AWS": "Cloud", "S3": "Cloud", "EC2": "Cloud", "Lambda": "Cloud", "Azure": "Cloud", "GCP": "Cloud",
    "Docker": "Cloud", "Kubernetes": "Cloud", "Terraform": "Cloud",
    "Spark": "Data", "Kafka": "Data", "Airflow": "Data",
    "Snowflake": "Data", "Databricks": "Data", "Tableau/PowerBI": "Data",
    "Control-M": "Cloud", "IICS": "Data", "Business Objects": "Data", "Perl": "App",
    # Roles as tech tags for JD emphasis
    "QA-Manual": "QA", "QA-Automation": "QA", "DevOps": "Cloud",
    "MLOps": "Data", "DataEngineer": "Data", "DataScientist": "Data",
}

# Role ↔ tech alignment for staffing and validity checks
_BASE_ROLE_TECH_ALIGNMENT = {
    "Backend": ["Java", "Python", "Node.js", ".NET", "SQL", "ASP.NET", "C#", "Ado.net", "MVC"],
    "Frontend": ["React", "Angular", "JavaScript", "HTML", "CSS", "AJAX"],
    "FullStack": ["Java", "Python", "Node.js", ".NET", "React", "Angular", "JavaScript", "HTML", "CSS", "AJAX"],
    "QA": ["QA-Manual", "QA-Automation"],
    "DevOps": ["AWS", "S3", "EC2", "Lambda", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "DevOps", "Control-M"],
    "DataEng": ["Spark", "Kafka", "Airflow", "Snowflake", "Databricks", "SQL", "SQL Server", "PostgreSQL", "OracleDB", "DataEngineer"],
    "DataSci": ["Python", "SQL", "DataScientist", "Tableau/PowerBI"],
    "MLE": ["MLOps", "Python", "AWS", "Azure", "GCP"],
    "Mobile": ["React"],
    "Architect": TECHNOLOGIES,
    "PM": TECHNOLOGIES,
    "UI/UX": [],
    "Support": ["SQL", "AWS", "Azure"],
}


DOMAIN_KEYWORDS = {
    "FinTech": ["fintech", "cards", "payments", "lending", "credit", "bank", "wallet"],
    "E-commerce": ["ecommerce", "e-commerce", "catalog", "checkout", "cart", "merchandis"],
    "Healthcare": ["health", "clinical", "patient", "claims", "hl7", "hipaa"],
    "Telecom": ["telecom", "5g", "network", "subscriber", "oss", "bss"],
    "Manufacturing": ["mes", "shop floor", "plm", "erp", "supply"],
    "Public Sector": ["public", "gov", "citizen", "procure"],
    "Insurance": ["policy", "premium", "underwriting", "claims","insurance"],
    "Retail": ["store", "pos", "loyalty", "merch"],
}


TECH_PATTERNS = [
    (re.compile(r"\bpython\b", re.I), "Python"),
    (re.compile(r"\bjava\b", re.I), "Java"),
    (re.compile(r"\.NET|dotnet", re.I), ".NET"),
    (re.compile(r"asp\.net|aspnet", re.I), "ASP.NET"),
    (re.compile(r"\bc#\b|csharp", re.I), "C#"),
    (re.compile(r"ado\.net|ado\s*net", re.I), "Ado.net"),
    (re.compile(r"\bmvc\b", re.I), "MVC"),
    (re.compile(r"node(\.js)?\b", re.I), "Node.js"),
    (re.compile(r"react\b", re.I), "React"),
    (re.compile(r"angular\b", re.I), "Angular"),
    (re.compile(r"javascript|js\b", re.I), "JavaScript"),
    (re.compile(r"\bhtml\b", re.I), "HTML"),
    (re.compile(r"\bcss\b", re.I), "CSS"),
    (re.compile(r"ajax\b", re.I), "AJAX"),
    (re.compile(r"\bsql\b", re.I), "SQL"),
    (re.compile(r"pl\s*/?\s*sql|plsql", re.I), "PL/SQL"),
    (re.compile(r"sql\s*server|mssql", re.I), "SQL Server"),
    (re.compile(r"postgres\w*|postgre\s*sql", re.I), "PostgreSQL"),
    (re.compile(r"oracle(\s*db|\s*database)?", re.I), "OracleDB"),
    (re.compile(r"\baws\b", re.I), "AWS"),
    (re.compile(r"\bamazon\s*s3\b|\baws\s*s3\b|\bs3\b", re.I), "S3"),
    (re.compile(r"\bec2\b", re.I), "EC2"),
    (re.compile(r"\blambda\b", re.I), "Lambda"),
    (re.compile(r"\bazure\b", re.I), "Azure"),
    (re.compile(r"\bgcp\b|google cloud", re.I), "GCP"),
    (re.compile(r"spark\b", re.I), "Spark"),
    (re.compile(r"kafka\b", re.I), "Kafka"),
    (re.compile(r"airflow\b", re.I), "Airflow"),
    (re.compile(r"docker\b", re.I), "Docker"),
    (re.compile(r"kubernetes|k8s\b", re.I), "Kubernetes"),
    (re.compile(r"terraform\b", re.I), "Terraform"),
    (re.compile(r"snowflake\b", re.I), "Snowflake"),
    (re.compile(r"databricks\b", re.I), "Databricks"),
    (re.compile(r"control[- ]?m\b", re.I), "Control-M"),
    (re.compile(r"informatica\s+(intelligent\s+cloud\s+services|cloud)\b|\biics\b", re.I), "IICS"),
    (re.compile(r"business\s*objects|sap\s*bo|sap\s*businessobjects|\bbo\b", re.I), "Business Objects"),
    (re.compile(r"\bperl\b", re.I), "Perl"),
    (re.compile(r"tableau|power\s?bi", re.I), "Tableau/PowerBI"),
    (re.compile(r"qa(\s|-)?manual", re.I), "QA-Manual"),
    (re.compile(r"qa(\s|-)?auto|selenium|cypress", re.I), "QA-Automation"),
    (re.compile(r"devops\b", re.I), "DevOps"),
    (re.compile(r"mlops\b", re.I), "MLOps"),
    (re.compile(r"data\s?engineer", re.I), "DataEngineer"),
    (re.compile(r"data\s?scientist", re.I), "DataScientist"),
]


_ROLE_DISPLAY_KEYWORDS: Dict[str, list[str]] = {
    "PM": ["product manager", "pm"],
    "Architect": ["solution architect", "tech lead", "architect"],
    "Backend": ["backend"],
    "Frontend": ["frontend"],
    "FullStack": ["full-stack", "full stack", "fullstack"],
    "DataEng": ["data engineer", "analytics engineer"],
    "DataSci": ["data scientist"],
    "MLE": ["ml engineer", "machine learning"],
    "DevOps": ["devops", "cloud engineer", "cloud", "mlops"],
    "QA": ["qa", "test automation"],
    "UI/UX": ["ux", "ui designer", "ui/ux", "ux/ui"],
    "Support": ["business analyst", "domain sme", "domain expert", "domain specialist"],
    "Mobile": ["mobile"],
}

_SKILL_ALIASES: Dict[str, str] = {
    "node": "Node.js",
    "nodejs": "Node.js",
    "js": "JavaScript",
    "javascript": "JavaScript",
    "ts": "TypeScript",
    "typescript": "TypeScript",
    "reactjs": "React",
    "react": "React",
    "py": "Python",
}


def _match_display_roles(display_name: str) -> set[str]:
    normalized = (display_name or "").lower()
    matches = {
        role
        for role, keywords in _ROLE_DISPLAY_KEYWORDS.items()
        if any(keyword in normalized for keyword in keywords)
    }
    if not matches:
        matches = {role for role in ROLE_FAMILIES if role.lower() in normalized}
    return matches


def _normalize_skill(value: str) -> str:
    token = (value or "").strip()
    if not token:
        return ""
    key = token.lower()
    if key in _SKILL_ALIASES:
        return _SKILL_ALIASES[key]
    for tech in TECHNOLOGIES:
        if tech.lower() == key:
            return tech
    return token


def _build_role_taxonomy() -> Dict[str, Dict[str, Any]]:
    raw = load_role_taxonomy_config()
    entries = raw.get("roles") or {}
    meta: Dict[str, Dict[str, Any]] = {}
    for role in ROLE_FAMILIES:
        base_skills = _BASE_ROLE_TECH_ALIGNMENT.get(role, [])
        meta[role] = {"skills": set(base_skills), "seniority_map": {}}
    for display_name, info in entries.items():
        matches = _match_display_roles(display_name)
        if not matches:
            continue
        skills = info.get("skills") or []
        seniority = info.get("seniority_map") or {}
        for role in matches:
            meta.setdefault(role, {"skills": set(), "seniority_map": {}})
            meta[role]["skills"].update({_normalize_skill(s) for s in skills if s})
            meta[role]["seniority_map"].update({k: v for k, v in seniority.items() if k})
    taxonomy: Dict[str, Dict[str, Any]] = {}
    for role, data in meta.items():
        cleaned = sorted({s for s in data["skills"] if s})
        taxonomy[role] = {"skills": cleaned, "seniority_map": data["seniority_map"]}
    return taxonomy


def _normalize_exp_split(exp_split: Any) -> Dict[str, float]:
    if not isinstance(exp_split, dict):
        return {}
    cleaned: Dict[str, float] = {}
    for level, weight in exp_split.items():
        if not level:
            continue
        try:
            value = float(weight)
        except Exception:
            continue
        normalized_level = level.strip().title()
        if not normalized_level:
            continue
        cleaned[normalized_level] = value
    total = sum(cleaned.values())
    if total <= 0:
        return {}
    return {lvl: val / total for lvl, val in cleaned.items()}


def _build_hiring_rule_targets() -> tuple[Dict[str, int], float, Dict[str, Dict[str, float]]]:
    raw = load_hiring_rule_config()
    rule_data = raw.get("hiring_rule") or {}
    tolerance = float(raw.get("tolerance_percent") or 0.0)
    totals: Dict[str, float] = {}
    total_sum = 0.0
    level_mix: Dict[str, Dict[str, float]] = {}
    for label, value in rule_data.items():
        percentage = None
        exp_split = None
        if isinstance(value, dict):
            percentage = value.get("percentage")
            exp_split = value.get("exp_split")
        else:
            percentage = value
        try:
            qty = float(percentage)
        except Exception:
            continue
        matches = _match_display_roles(label)
        if not matches:
            continue
        share = qty / len(matches)
        for role in matches:
            totals[role] = totals.get(role, 0.0) + share
            if exp_split:
                normalized = _normalize_exp_split(exp_split)
                if normalized and role not in level_mix:
                    level_mix[role] = normalized
        total_sum += qty
    target_total = int(round(total_sum))
    floors = {role: int(math.floor(val)) for role, val in totals.items()}
    residual = target_total - sum(floors.values())
    sorted_roles = sorted(
        totals.items(),
        key=lambda item: (item[1] - math.floor(item[1])),
        reverse=True,
    )
    for role, _ in sorted_roles:
        if residual <= 0:
            break
        floors[role] = floors.get(role, 0) + 1
        residual -= 1
    if residual < 0:
        for role, _ in reversed(sorted_roles):
            if residual >= 0:
                break
            if floors.get(role, 0) > 0:
                floors[role] -= 1
                residual += 1
    final = {role: count for role, count in floors.items() if count > 0}
    return final, tolerance, level_mix


ROLE_TAXONOMY = _build_role_taxonomy()
ROLE_TECH_ALIGNMENT = {role: info["skills"] for role, info in ROLE_TAXONOMY.items()}
_HIRING_RULE_TARGETS, _HIRING_RULE_TOLERANCE, _HIRING_RULE_LEVEL_MIX = _build_hiring_rule_targets()



def infer_domain(text: str) -> str | None:
    t = text.lower()
    for domain, keys in DOMAIN_KEYWORDS.items():
        for k in keys:
            if k in t:
                return domain
    return None


def extract_technologies(text: str) -> list[str]:
    hits = []
    for pat, tech in TECH_PATTERNS:
        if pat.search(text):
            hits.append(tech)
    # De-dup while preserving order
    seen = set()
    out = []
    for h in hits:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out


def min_exp_near(text: str, tech: str) -> float | None:
    # Look for patterns like "3+ years in Spark" or "at least 2 years with Kafka"
    t = text
    idx = t.lower().find(tech.lower().split('/')[0])
    window = t[max(0, idx - 40): idx + 60] if idx != -1 else t
    m = re.search(r"(\d+)(\+)?\s*(\+|plus)?\s*(years|yrs)", window, re.I)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None


def map_role_family_from_tech(tech: str) -> str | None:
    if not tech:
        return None
    normalized = tech.strip().lower()
    for role, techs in ROLE_TECH_ALIGNMENT.items():
        for candidate in techs:
            if normalized == candidate.strip().lower():
                return role
    return None


def role_level_mix_default(role: str) -> dict:
    if role in _HIRING_RULE_LEVEL_MIX:
        return dict(_HIRING_RULE_LEVEL_MIX[role])
    # A rough default mix that sums to 1.0
    if role in ("PM", "Architect"):
        return {"Senior": 0.5, "Lead": 0.4, "Principal": 0.1}
    if role in ("Backend", "Frontend", "FullStack", "DataEng", "DataSci", "DevOps", "MLE"):
        return {"Junior": 0.15, "Mid": 0.35, "Senior": 0.35, "Lead": 0.12, "Principal": 0.03}
    if role in ("QA", "Support"):
        return {"Junior": 0.25, "Mid": 0.45, "Senior": 0.25, "Lead": 0.05}
    if role in ("UI/UX", "Mobile"):
        return {"Junior": 0.2, "Mid": 0.4, "Senior": 0.3, "Lead": 0.1}
    return {"Mid": 0.5, "Senior": 0.5}


def region_cost_multiplier(region: str | None) -> float:
    if region == "NA":
        return 1.8
    if region == "EU":
        return 1.5
    if region == "APAC":
        return 1.2
    if region == "India":
        return 1.0
    return 1.3


def get_role_taxonomy() -> Dict[str, Dict[str, Any]]:
    return {
        role: {
            "skills": list(info.get("skills", [])),
            "seniority_map": dict(info.get("seniority_map", {})),
        }
        for role, info in ROLE_TAXONOMY.items()
    }


def get_hiring_rule_targets() -> Dict[str, int]:
    return dict(_HIRING_RULE_TARGETS)


def get_hiring_rule_tolerance_percent() -> float:
    return _HIRING_RULE_TOLERANCE

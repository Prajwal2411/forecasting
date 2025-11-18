import json
from typing import Any, Dict, List

from .utils.ontology import ROLE_FAMILIES, REGIONS, LEVELS, role_level_mix_default
from .utils.tech_extractor import _priority_for, _canonical_tech
from .llm_client import LLMClient


def _clamp_priority(v: Any) -> int:
    try:
        iv = int(v)
    except Exception:
        iv = 2
    if iv >= 3:
        return 3
    return 2


def parse_jd_llm(text: str, tech_section: str | None = None) -> Dict[str, Any]:
    client = LLMClient()
    if not client.available():
        raise RuntimeError("LLM unavailable")

    schema_hint = {
        "project_title": "string",
        "tech_requirements": [
            {"technology": "string", "priority": "2|3", "min_exp_years": "number"}
        ],
        "roles_needed": [
            {
                "role_family": "one of ROLE_FAMILIES",
                "count": "integer",
                "level_mix": {lvl: "0..1" for lvl in LEVELS},
            }
        ],
        "start_quarter": "YYYYQn",
        "end_quarter": "YYYYQn",
        "budget_ceiling": "number",
        "constraints": {"region": "India|NA|EU|APAC|optional"},
    }
    sys = (
        "You extract structured hiring signals from a job description. "
        "Return ONLY strict JSON with the fields in the provided schema: no prose. "
        "Fill role level_mix to sum to 1.0 if possible."
    )
    user = (
        "Schema hint: "
        + json.dumps(schema_hint)
        + "\nJD text:\n" + (text or "")
        + ("\nStrict tech section:\n" + tech_section if tech_section else "")
    )
    data = client.chat_json(sys, user)

    out: Dict[str, Any] = {}
    out["project_title"] = (data.get("project_title") or "JD").strip()[:120]

    techs_in: List[Dict[str, Any]] = data.get("tech_requirements") or []
    techs_out: List[Dict[str, Any]] = []
    for t in techs_in:
        raw = str(t.get("technology") or "").strip()
        can = _canonical_tech(raw) or raw
        if not can:
            continue
        pr = _clamp_priority(t.get("priority")) if t.get("priority") is not None else _priority_for(can)
        try:
            yrs = float(t.get("min_exp_years", 2.0))
        except Exception:
            yrs = 2.0
        techs_out.append({"technology": can, "priority": pr, "min_exp_years": yrs})
    out["tech_requirements"] = techs_out

    roles_in: List[Dict[str, Any]] = data.get("roles_needed") or []
    roles_out: List[Dict[str, Any]] = []
    for r in roles_in:
        rf = str(r.get("role_family") or "").strip()
        if rf not in ROLE_FAMILIES:
            continue
        try:
            cnt = max(1, int(r.get("count", 1)))
        except Exception:
            cnt = 1
        mix = r.get("level_mix") or role_level_mix_default(rf)
        # keep only known levels and renorm
        mix2 = {k: float(v) for k, v in mix.items() if k in LEVELS}
        s = sum(mix2.values()) or 1.0
        mix2 = {k: v / s for k, v in mix2.items()}
        roles_out.append({"role_family": rf, "count": cnt, "level_mix": mix2})
    out["roles_needed"] = roles_out

    def _norm_q(s: str | None) -> str | None:
        if not s:
            return None
        s = s.upper().replace(" ", "")
        return s if len(s) >= 6 and s.count("Q") == 1 else None

    out["start_quarter"] = _norm_q(data.get("start_quarter"))
    out["end_quarter"] = _norm_q(data.get("end_quarter"))
    try:
        out["budget_ceiling"] = float(data.get("budget_ceiling")) if data.get("budget_ceiling") is not None else None
    except Exception:
        out["budget_ceiling"] = None

    cons = data.get("constraints") or {}
    reg = cons.get("region") if isinstance(cons, dict) else None
    if reg not in REGIONS:
        reg = None
    out["constraints"] = {"region": reg} if reg else {}

    return out


def propose_team_shape_llm(jd_text: str, tech_requirements: list[dict] | None, desired_team_size: int) -> dict:
    """Ask the LLM for a realistic team shape given the JD and desired team size.
    Enforces hard business rules after response: PM<=1, Architect<=2, non-negative integers,
    and total equals desired_team_size. Returns a mapping role->count.
    """
    client = LLMClient()
    if not client.available():
        raise RuntimeError("LLM unavailable")

    role_list = list(ROLE_FAMILIES)
    sys = (
        "You design realistic team compositions for software/data projects. "
        "Return ONLY compact JSON: {\"roles\": {role: count, ...}} with no extra text. "
        "Rules: At most 1 PM, at most 2 Architects; include QA and DevOps at ~1:6 and ~1:8 ratio to engineers respectively. "
        "Distribute engineers across relevant roles based on the technologies and domain. "
        "Counts must be integers >=0 and sum to the requested team size. Allowed roles are: "
        + ", ".join(role_list)
    )
    tech_json = json.dumps(tech_requirements or [])
    user = (
        f"Desired team size: {desired_team_size}\n"
        f"JD text:\n{jd_text}\n"
        f"Tech requirements (normalized): {tech_json}\n"
        "Return strictly the JSON schema specified."
    )
    data = client.chat_json(sys, user)
    roles = (data or {}).get("roles") or {}
    # Clamp and clean
    out: dict[str, int] = {}
    total = 0
    for k, v in roles.items():
        if k not in ROLE_FAMILIES:
            continue
        try:
            iv = max(0, int(v))
        except Exception:
            iv = 0
        out[k] = iv
        total += iv
    # Enforce caps
    if out.get("PM", 0) > 1:
        total -= out["PM"] - 1
        out["PM"] = 1
    if out.get("Architect", 0) > 2:
        total -= out["Architect"] - 2
        out["Architect"] = 2
    # Rebalance to match desired size
    def add_to_top(keys: list[str], n: int):
        i = 0
        while n > 0 and keys:
            k = keys[i % len(keys)]
            out[k] = out.get(k, 0) + 1
            n -= 1
            i += 1

    def take_from_largest(keys: list[str], n: int):
        while n > 0:
            if not keys:
                break
            k = max(keys, key=lambda kk: out.get(kk, 0))
            if out.get(k, 0) <= 0:
                keys.remove(k)
                continue
            out[k] -= 1
            n -= 1

    if total < desired_team_size:
        # Add to engineer roles as default
        eng = [r for r in ROLE_FAMILIES if r in ("Backend", "Frontend", "FullStack", "DataEng", "DataSci", "MLE", "Mobile")]
        add_to_top(eng, desired_team_size - total)
    elif total > desired_team_size:
        # Remove from largest engineer roles first
        eng = [r for r in ROLE_FAMILIES if r in ("Backend", "Frontend", "FullStack", "DataEng", "DataSci", "MLE", "Mobile")]
        take_from_largest(eng, total - desired_team_size)

    return out

import json
import re
from typing import Any, Dict, List, Tuple

from .utils.ontology import (
    REGIONS,
    infer_domain,
    map_role_family_from_tech,
    role_level_mix_default,
)
from .utils.tech_extractor import parse_strict_tech_roles, _priority_for
from .utils.ontology import extract_technologies, min_exp_near


def normalize_quarter(token: str) -> str | None:
    # Accept formats: YYYYQn, YYYY Qn, Qn YYYY, Qn-YYYY
    t = token.strip()
    m = re.match(r"(\d{4})\s*-?\s*Q\s*([1-4])$", t, re.I)
    if m:
        return f"{m.group(1)}Q{m.group(2)}"
    m = re.match(r"^Q\s*([1-4])\s*-?\s*(\d{4})$", t, re.I)
    if m:
        return f"{m.group(2)}Q{m.group(1)}"
    return None


def parse_jd(
    text: str,
    tech_section: str | None = None,
) -> Dict[str, Any]:
    # Prefer LLM-driven parser; on success, enrich with free-text extras
    try:
        from .jd_parser_llm import parse_jd_llm
        llm_parsed = parse_jd_llm(text, tech_section=tech_section)
        if llm_parsed:
            tr = llm_parsed.get("tech_requirements") or []
            have = {d.get("technology") for d in tr if d.get("technology")}
            extras = extract_technologies(text)
            for t in extras:
                if t in have:
                    continue
                yrs = min_exp_near(text, t) or 2.0
                tr.append({
                    "technology": t,
                    "priority": _priority_for(t),
                    "min_exp_years": float(yrs),
                })
            llm_parsed["tech_requirements"] = tr
            return llm_parsed
    except Exception:
        pass
    title = None
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        title = lines[0][:120]

    domain = infer_domain(text)

    # Helpers for additional extraction
    def find_overall_experience(txt: str) -> float | None:
        m = re.search(r"(\d{1,2})(?:\+|\s*\+?\s*years?)\s+(?:overall|total)?\s*experience", txt, re.I)
        if not m:
            m = re.search(r"overall\s*experience\s*(\d{1,2})\+?\s*years?", txt, re.I)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return None
        return None

    def find_duration_months(txt: str) -> int | None:
        m = re.search(r"(\d{1,2})\s*(?:months|mos)\b", txt, re.I)
        if m:
            return int(m.group(1))
        m = re.search(r"(\d)\s*(?:years?|yrs?)\b", txt, re.I)
        if m:
            return int(m.group(1)) * 12
        return None

    def find_team_size(txt: str) -> int | None:
        m = re.search(r"team\s*(?:size|includes|of)\s*(\d{1,3})", txt, re.I)
        if m:
            return int(m.group(1))
        m = re.search(r"(\d{1,2})\s*(?:engineers|developers)\b", txt, re.I)
        if m:
            return int(m.group(1))
        return None

    def extract_role_counts(txt: str) -> Dict[str, int]:
        patterns = {
            "PM": r"(\d+)\s*(?:PM|project\s*manager)s?",
            "Architect": r"(\d+)\s*architects?",
            "Backend": r"(\d+)\s*(?:backend|server[- ]side)\b",
            "Frontend": r"(\d+)\s*(?:frontend|ui)\b",
            "FullStack": r"(\d+)\s*(?:full[- ]?stack)\b",
            "QA": r"(\d+)\s*(?:qa|test(?:er|ing)?)\b",
            "DevOps": r"(\d+)\s*(?:devops|sre)s?\b",
            "DataEng": r"(\d+)\s*(?:data\s*engineers?)\b",
            "DataSci": r"(\d+)\s*(?:data\s*scientists?)\b",
        }
        out: Dict[str, int] = {}
        for role, pat in patterns.items():
            m = re.search(pat, txt, re.I)
            if m:
                try:
                    out[role] = out.get(role, 0) + int(m.group(1))
                except Exception:
                    pass
        # Ranges like 4-6 engineers
        m = re.search(r"(\d+)\s*[-–]\s*(\d+)\s*(engineers|developers)", txt, re.I)
        if m:
            try:
                lo = int(m.group(1)); hi = int(m.group(2))
                out["Backend"] = max(out.get("Backend", 0), (lo + hi) // 2)
            except Exception:
                pass
        return out

    def extract_cloud_pref(txt: str) -> List[str]:
        prefs = []
        for c in ["AWS", "Azure", "GCP"]:
            if re.search(rf"\b{c}\b", txt, re.I):
                prefs.append(c)
        return prefs

    def extract_methodologies(txt: str) -> List[str]:
        items = []
        for w in ["Agile", "Scrum", "Kanban", "CI/CD", "DevOps", "Microservices"]:
            if re.search(rf"\b{re.escape(w)}\b", txt, re.I):
                items.append(w)
        return items

    def extract_certs(txt: str) -> List[str]:
        pats = [r"PMP", r"CSM", r"AZ-\d{3}", r"AWS\s+Certified[\w\s-]*", r"GCP\s+Professional[\w\s-]*"]
        out = []
        for p in pats:
            if re.search(p, txt, re.I):
                out.append(re.search(p, txt, re.I).group(0))
        return out

    # Strict Tech Stack parsing (preferred when provided)
    tech_requirements, role_counts_strict = parse_strict_tech_roles(tech_section or "")
    # Always enrich with free-text technology extraction to capture extras
    techs_free = extract_technologies(text)
    have = {d.get("technology") for d in tech_requirements}
    for t in techs_free:
        if t in have:
            continue
        have.add(t)
        yrs = min_exp_near(text, t) or 2.0
        tech_requirements.append({
            "technology": t,
            "priority": _priority_for(t),
            "min_exp_years": float(yrs),
        })

    # Optional NLP enhancement (spaCy) — if available, prefer its results and replace earlier ones
    def nlp_enhance(txt: str) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        try:
            import spacy  # type: ignore
            from spacy.matcher import PhraseMatcher, Matcher  # type: ignore
        except Exception:
            return [], {}

        # Use spaCy pipeline if available, otherwise a blank English model
        nlp = spacy.load("en_core_web_sm") if getattr(spacy.util, "is_package", lambda *_: False)("en_core_web_sm") else spacy.blank("en")
        doc = nlp(txt)

        # Phrase match technologies using ontology terms + aliases from config
        try:
            from .utils.ontology import tech_phrase_lexicon, canonicalize_tech  # type: ignore
        except Exception:
            # Fallback: no lexicon helpers present
            tech_phrase_lexicon = lambda: []  # type: ignore
            canonicalize_tech = lambda s: s  # type: ignore

        pm = PhraseMatcher(nlp.vocab, attr="LOWER")
        tech_patterns = [nlp.make_doc(t) for t in tech_phrase_lexicon()]
        if tech_patterns:
            pm.add("TECH", tech_patterns)

        # Years-of-experience patterns per tech
        matcher = Matcher(nlp.vocab)
        matcher.add(
            "TECH_YEARS_A",
            [[{"LIKE_NUM": True}, {"LOWER": {"IN": ["year", "years", "yrs", "yr"]}}, {"LOWER": {"IN": ["of", "+"]}, "OP": "?"}, {"IS_ALPHA": True}]],
        )
        matcher.add(
            "TECH_YEARS_B",
            [[{"IS_ALPHA": True}, {"LIKE_NUM": True}, {"LOWER": {"IN": ["year", "years", "yrs", "yr"]}}]],
        )

        spans = pm(doc, as_spans=True) if tech_patterns else []
        found_techs = set([canonicalize_tech(s.text) for s in spans])
        specific_dbs = {"SQL Server", "PostgreSQL", "OracleDB"}
        if found_techs & specific_dbs:
            found_techs.discard("SQL")
        tech_map: Dict[str, float] = {}
        for s in spans:
            start = max(0, s.start - 4)
            end = min(len(doc), s.end + 4)
            win = doc[start:end]
            years = None
            for tok in win:
                if tok.like_num:
                    try:
                        v = float(tok.text)
                        if 0 < v < 50:
                            years = v
                            break
                    except Exception:
                        pass
            tech_map[canonicalize_tech(s.text)] = years if years is not None else 2.0

        from .utils.ontology import ROLE_FAMILIES
        pm_roles = PhraseMatcher(nlp.vocab, attr="LOWER")
        pm_roles.add("ROLE", [nlp.make_doc(r) for r in ROLE_FAMILIES])
        role_counts: Dict[str, int] = {}
        for s in pm_roles(doc, as_spans=True):
            role_counts[s.text] = role_counts.get(s.text, 0) + 1

        tr: List[Dict[str, Any]] = []
        for t in sorted(found_techs):
            prio = 3 if t in ("Spark", "Kafka", "Snowflake", "Databricks", "AWS", "Azure", "GCP") else 2
            tr.append({"technology": t, "priority": prio, "min_exp_years": float(tech_map.get(t, 2.0))})
        return tr, role_counts

    # Optional NLP enhancement for roles only; techs come from strict section
    nlp_tr, nlp_role_counts = nlp_enhance(text)
    extraction_method = "strict_tech_section"
    role_counts_text = role_counts_strict if role_counts_strict else (nlp_role_counts or extract_role_counts(text))

    # Roles needed from strict counts and light text fallback
    role_counts: Dict[str, int] = dict(role_counts_text) if role_counts_text else {}
    # Fallback: infer unique roles from technologies (one per role). Counts will be
    # scaled later based on desired team size in the app UI.
    if not role_counts and tech_requirements:
        inferred_roles = set()
        for tr in tech_requirements:
            tname = tr.get("technology")
            rf = map_role_family_from_tech(tname) if tname else None
            if rf:
                inferred_roles.add(rf)
        for rf in inferred_roles:
            role_counts[rf] = 1
    # If FullStack present, split to Backend/Frontend
    if role_counts.get("FullStack"):
        fs = role_counts.pop("FullStack")
        be = (fs + 1) // 2
        fe = fs - be
        role_counts["Backend"] = role_counts.get("Backend", 0) + be
        role_counts["Frontend"] = role_counts.get("Frontend", 0) + max(1, fe)
    # Always ensure PM and at least one QA
    role_counts["PM"] = max(1, role_counts.get("PM", 0))
    role_counts["QA"] = max(1, role_counts.get("QA", 0))

    roles_needed = []
    for rf, cnt in sorted(role_counts.items(), key=lambda x: -x[1]):
        mix = role_level_mix_default(rf)
        roles_needed.append({
            "role_family": rf,
            "count": max(1, cnt),
            "level_mix": mix,
        })

    # Budget ceiling
    budget_ceiling = None
    m = re.search(r"budget[^\d]*(\d[\d,\.]+)", text, re.I)
    if m:
        try:
            budget_ceiling = float(m.group(1).replace(",", ""))
        except Exception:
            budget_ceiling = None

    # Quarter window
    start_quarter = None
    end_quarter = None
    qs = re.findall(r"(?:\d{4}\s*-?\s*Q\s*[1-4]|Q\s*[1-4]\s*-?\s*\d{4})", text, re.I)
    if qs:
        sq = normalize_quarter(qs[0])
        if sq:
            start_quarter = sq.replace(" ", "")
    if len(qs) > 1:
        eq = normalize_quarter(qs[1])
        if eq:
            end_quarter = eq.replace(" ", "")

    # Region constraints
    region = None
    for r in REGIONS:
        if re.search(rf"\b{re.escape(r)}\b", text, re.I):
            region = r
            break

    constraints = {}
    if region:
        constraints["region"] = region
    onsite = re.search(r"on[- ]?site|on\s*site", text, re.I)
    if onsite:
        constraints["onsite_percent"] = 50

    # Additional signals
    overall_exp = find_overall_experience(text + ("\n" + (tech_section or "")))
    duration_months = find_duration_months(text)
    team_size = find_team_size(text)
    cloud_pref = extract_cloud_pref(text)
    methods = extract_methodologies(text)
    certs = extract_certs(text)

    return {
        "project_title": title or "JD",
        "domain": domain,
        "tech_requirements": tech_requirements,
        "roles_needed": roles_needed,
        "start_quarter": start_quarter,
        "end_quarter": end_quarter,
        "budget_ceiling": budget_ceiling,
        "constraints": constraints,
        "extraction_method": extraction_method,
        "overall_experience_years": overall_exp,
        "duration_months": duration_months,
        "team_size": team_size,
        "cloud_preference": cloud_pref,
        "methodologies": methods,
        "certifications": certs,
    }


if __name__ == "__main__":
    import sys
    txt = sys.stdin.read()
    parsed = parse_jd(txt)
    print(json.dumps(parsed, indent=2))

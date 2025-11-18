from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
HIRING_RULE_PATH = ROOT / "hiring_rule.yml"
ROLE_TAXONOMY_PATH = ROOT / "roles.yml"

_CACHE: Dict[str, Any] = {}


def _safe_load(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    if path in _CACHE:
        return _CACHE[path]
    if yaml is None:
        _CACHE[path] = default
        return default
    if not path.exists():
        _CACHE[path] = default
        return default
    try:
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(content, dict):
            _CACHE[path] = content
            return content
    except Exception:
        pass
    _CACHE[path] = default
    return default


def load_hiring_rule_config() -> Dict[str, Any]:
    """Return the raw hiring rule configuration."""
    return _safe_load(HIRING_RULE_PATH, {"hiring_rule": {}, "tolerance_percent": 0})


def load_role_taxonomy_config() -> Dict[str, Any]:
    """Return the raw role taxonomy configuration."""
    return _safe_load(ROLE_TAXONOMY_PATH, {"roles": {}})

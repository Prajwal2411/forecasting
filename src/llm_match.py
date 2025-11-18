from typing import Iterable, List, Set

from .llm_client import LLMClient


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    s = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return s / (na * nb)


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def score_match(jd_text: str, cand_text: str, jd_tags: Iterable[str] | None = None, cand_tags: Iterable[str] | None = None) -> float:
    # Try embeddings; fallback to simple Jaccard on tags
    client = LLMClient()
    if client.available():
        try:
            emb = client.embed([jd_text, cand_text])
            return float(_cosine(emb[0], emb[1]))
        except Exception:
            pass
    a = {t.lower() for t in jd_tags or []}
    b = {t.lower() for t in cand_tags or []}
    return float(jaccard(a, b))


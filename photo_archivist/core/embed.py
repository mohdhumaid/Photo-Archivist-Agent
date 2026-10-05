"""Step 8: Embed interface — text hash vectors now; MiniLM/CLIP plug in later."""
from __future__ import annotations
import hashlib
import math


def text_vector(text: str, dim: int = 128) -> list[float]:
    vec = [0.0] * dim
    for tok in (text or "").lower().split():
        h = int(hashlib.sha256(tok.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    n = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / n for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    try:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (na * nb)
    except Exception:
        return 0.0

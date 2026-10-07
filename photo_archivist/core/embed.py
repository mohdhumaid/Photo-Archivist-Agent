"""Step 8: Embeddings — local deterministic hash ONLY.

No model downloads: no sentence-transformers /
Hugging Face downloads, so vectors stay bag-of-words hashes (128-dim, hash-based
similarity only). Richer captions/tags come from the vision-LLM when enabled.
"""
from __future__ import annotations
import hashlib
import math

HASH_DIM = 128


def hash_vector(text: str, dim: int = HASH_DIM) -> list[float]:
    """Deterministic bag-of-words hash — works everywhere, no AI semantics."""
    vec = [0.0] * dim
    for tok in (text or "").lower().split():
        h = int(hashlib.sha256(tok.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    n = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / n for v in vec]


def text_vector(text: str, dim: int = HASH_DIM, backend: str = "hash") -> list[float]:
    """Local hash vectors only — backend argument kept for config compatibility."""
    return hash_vector(text, dim)


def cosine(a: list[float], b: list[float]) -> float:
    try:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (na * nb)
    except Exception:
        return 0.0

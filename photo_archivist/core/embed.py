"""Step 8: Embeddings — LIVE sentence-transformers when installed, hash fallback.

backend: "auto"  -> sentence-transformers if importable, else hash (default)
         "sbert" -> require sentence-transformers (raises when missing)
         "hash"  -> force deterministic bag-of-words hash (128-dim)
Switching backends changes vector dimensions (384 vs 128) — rebuild the index
(delete index.db + re-scan) so all stored vectors share one dimension.
"""
from __future__ import annotations
import hashlib
import math

HASH_DIM = 128
ST_DIM = 384
_st_model = None
_st_failed = False


def _sbert_model():
    """Lazily load all-MiniLM-L6-v2 (~90 MB first download, cached afterwards)."""
    global _st_model, _st_failed
    if _st_failed:
        return None
    if _st_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _st_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _st_failed = True
    return _st_model


def hash_vector(text: str, dim: int = HASH_DIM) -> list[float]:
    """Deterministic bag-of-words hash — works everywhere, no AI semantics."""
    vec = [0.0] * dim
    for tok in (text or "").lower().split():
        h = int(hashlib.sha256(tok.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    n = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / n for v in vec]


def text_vector(text: str, dim: int = HASH_DIM, backend: str = "auto") -> list[float]:
    if backend in ("auto", "sbert"):
        m = _sbert_model()
        if m is not None:
            return [float(x) for x in m.encode(text or "", normalize_embeddings=True)]
        if backend == "sbert":
            raise RuntimeError(
                "sentence-transformers unavailable — run `pip install -e .[ai-local]` "
                "or set embeddings.backend: auto")
    return hash_vector(text, dim)


def cosine(a: list[float], b: list[float]) -> float:
    try:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (na * nb)
    except Exception:
        return 0.0

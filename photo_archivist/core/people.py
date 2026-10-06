"""Step 6: People — region-tag ground truth (0.99) seeds library; local match above threshold."""
from __future__ import annotations


def region_names(raw: dict) -> list[dict]:
    """Pull MWG/MP region names + PersonInImage as ground truth."""
    people: list[dict] = []
    for k, v in (raw or {}).items():
        kl = k.lower()
        if "regioninfo" in kl or "personinimage" in kl or "lastkeywordxmp" in kl:
            vals = v if isinstance(v, list) else [v]
            for item in vals:
                if isinstance(item, dict):
                    name = item.get("Name") or item.get("PersonDisplayName") or item.get("name")
                    box = item.get("Rectangle") or item.get("box")
                    if name:
                        people.append({"name": str(name), "source": "xmp_mwg_region",
                                       "confidence": 0.99, "box": box})
                elif isinstance(item, str) and item.strip():
                    people.append({"name": item.strip(), "source": "xmp_personinimage",
                                   "confidence": 0.99})
    # also plain string fields
    for k, v in (raw or {}).items():
        if k.split(":")[-1] in ("PersonInImage",) and isinstance(v, str) and v.strip():
            if not any(p["name"] == v.strip() for p in people):
                people.append({"name": v.strip(), "source": "xmp_personinimage",
                               "confidence": 0.99})
    return people


def match_local(embedding: list | None, library: dict, threshold: float = 0.45, min_margin: float = 0.05) -> dict | None:
    """Cosine match against local people library with margin separation verification.

    Returns match dict or None when confidence is below threshold or ambiguous.
    """
    if not embedding or not library:
        return None
    import math
    scores: list[tuple[str, float]] = []
    na = math.sqrt(sum(a * a for a in embedding)) or 1.0

    for name, vec in library.items():
        if len(vec) != len(embedding):
            continue
        try:
            dot = sum(a * b for a, b in zip(embedding, vec))
            nb = math.sqrt(sum(b * b for b in vec)) or 1.0
            s = dot / (na * nb)
            scores.append((name, s))
        except Exception:
            continue

    if not scores:
        return None

    scores.sort(key=lambda x: x[1], reverse=True)
    best_name, best_s = scores[0]
    second_s = scores[1][1] if len(scores) > 1 else 0.0

    # Margin separation check: avoid ambiguous / near-tied matches
    margin = best_s - second_s
    if best_s >= threshold and (len(scores) == 1 or margin >= min_margin):
        return {
            "name": best_name,
            "source": "face_match_local",
            "confidence": round(float(best_s), 3),
        }
    return None


def load_face_library(path: str | None) -> dict:
    """Face embeddings library {name: [512 floats]} from a local JSON file.

    Missing/invalid file -> empty dict (face matching simply stays off).
    """
    import json
    import os
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            lib = json.load(f) or {}
        return {k: v for k, v in lib.items() if isinstance(v, list) and v}
    except Exception:
        return {}

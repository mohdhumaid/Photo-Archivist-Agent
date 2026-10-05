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


def match_local(embedding: list | None, library: dict, threshold: float = 0.60) -> dict | None:
    """Cosine match against local people library. Returns match or None."""
    if not embedding or not library:
        return None
    import math
    best, best_s = None, -1.0
    for name, vec in library.items():
        try:
            dot = sum(a * b for a, b in zip(embedding, vec))
            na = math.sqrt(sum(a * a for a in embedding)) or 1.0
            nb = math.sqrt(sum(b * b for b in vec)) or 1.0
            s = dot / (na * nb)
        except Exception:
            continue
        if s > best_s:
            best, best_s = name, s
    if best and best_s >= threshold:
        return {"name": best, "source": "face_match_local", "confidence": round(float(best_s), 3)}
    return None

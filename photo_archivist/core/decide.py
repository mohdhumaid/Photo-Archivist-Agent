"""Step 10: Folder decision (S5). score = .5 vec + .3 tag + .2 people."""
from __future__ import annotations
from .embed import cosine


def score_folder(file_vec: list, file_tags: set, file_people: set, profile: dict) -> float:
    v = cosine(file_vec or [], profile.get("centroid") or [])
    ptags = set(profile.get("tags") or [])
    ppeople = set(profile.get("people") or [])
    tag_overlap = len(file_tags & ptags) / max(1, len(file_tags | ptags))
    ppl_overlap = len(file_people & ppeople) / max(1, len(file_people | ppeople))
    score = 0.50 * v + 0.30 * tag_overlap + 0.20 * ppl_overlap
    # hard metadata corroboration boost
    if profile.get("hard_match"):
        score = min(1.0, score + 0.10)
    return round(score, 3)


def decide(file_vec: list, file_tags: set, file_people: set, profiles: list,
           promote: float = 0.80, review_low: float = 0.60) -> dict:
    ranked = sorted(
        ((p.get("name", ""), score_folder(file_vec, file_tags, file_people, p)) for p in profiles),
        key=lambda t: t[1], reverse=True,
    )
    if not ranked:
        return {"action": "new_folder", "score": 0.0}
    name, s = ranked[0]
    if s >= promote:
        return {"action": "file", "folder": name, "score": s}
    if s >= review_low:
        second = ranked[1] if len(ranked) > 1 else (None, 0.0)
        return {"action": "review", "candidates": [list(ranked[0]), list(second)], "score": s}
    return {"action": "new_folder", "score": s}

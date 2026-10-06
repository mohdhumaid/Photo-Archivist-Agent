"""Dry-run report (S9): coverage stats, tags, folders, questions, duplicates."""
from __future__ import annotations
from collections import Counter


def build_report(records: list[dict], decisions: list[dict],
                 unnamed_face_ask_n: int = 3) -> dict:
    n = len(records) or 1
    has_exif = sum(1 for r in records if (r.get("raw_metadata") or {}).get("exif"))
    has_gps = sum(1 for r in records if (r.get("place") or {}).get("gps"))
    has_people = sum(1 for r in records if r.get("people"))
    has_faces = sum(1 for r in records if r.get("face_boxes"))
    has_kw = sum(1 for r in records if r.get("tags"))
    stripped = sum(1 for r in records if r.get("metadata_status") == "stripped")
    dups: dict = {}
    for r in records:
        dups.setdefault(r.get("sha256"), []).append(r.get("source_path"))
    dup_groups = {k: v for k, v in dups.items() if len(v) > 1}
    questions: list[str] = []
    for r, d in zip(records, decisions):
        if d.get("action") == "review":
            questions.append(f"REVIEW {r['source_path']} -> {d.get('candidates')}")
        if (r.get("taken_at_confidence") or 0) < 0.4:
            questions.append(f"DATE-ONLY-MTIME {r['source_path']}: only unreliable date")
        if any("pii" in str(f) or "pan_" in str(f) or "aadhaar" in str(f) for f in r.get("pii_flags", [])):
            questions.append(f"PII-FLAG {r['source_path']}: {r.get('pii_flags')}")
        n_faces = len(r.get("face_boxes") or [])
        if n_faces >= unnamed_face_ask_n and not r.get("people"):
            questions.append(
                f"UNNAMED-FACE {r['source_path']}: {n_faces} faces detected, none named")
    tag_counts = Counter(t for r in records for t in (r.get("tags") or []))
    return {
        "files": len(records),
        "coverage": {"exif_pct": round(100 * has_exif / n, 1), "gps_pct": round(100 * has_gps / n, 1),
                     "people_pct": round(100 * has_people / n, 1),
                     "faces_pct": round(100 * has_faces / n, 1),
                     "keywords_pct": round(100 * has_kw / n, 1),
                     "stripped_pct": round(100 * stripped / n, 1)},
        "top_tags": tag_counts.most_common(20),
        "decisions": Counter(d.get("action") for d in decisions),
        "duplicates": dup_groups, "questions": questions,
    }

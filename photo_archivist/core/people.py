"""Step 6: People — region-tag ground truth (0.99) seeds library; local match above threshold.

MULTI-FACE RULE: one image can hold many faces. Every detected face gets its
own box + identity attempt, and ALL results are tagged on the record:

  persons_for_faces(detections, raw, library, ...) -> (persons, unnamed_count)

Priority per face: MWG/MP region name (0.99, ground truth) > local face-match
(threshold) > vision people_hints (<=0.5, never auto-filed alone) >
"Unknown Person <n>" placeholder (0.2) so _Review/ can ask for a name once.
"""
from __future__ import annotations
import re


def _norm_box(item: dict):
    """Normalise MWG/MP rectangles (various spellings) to [x,y,w,h] floats."""
    for k in ("Rectangle", "rect", "Box", "box", "RegionRectangle",
              "Area", "Bounds"):
        b = item.get(k)
        if b is None:
            continue
        if isinstance(b, dict):
            try:
                return [float(b.get("X", b.get("x", 0))),
                        float(b.get("Y", b.get("y", 0))),
                        float(b.get("W", b.get("w", b.get("Width", 0)))),
                        float(b.get("H", b.get("h", b.get("Height", 0))))]
            except Exception:
                continue
        if isinstance(b, (list, tuple)) and len(b) >= 4:
            try:
                return [float(x) for x in b[:4]]
            except Exception:
                continue
        if isinstance(b, str):
            nums = re.findall(r"[-+]?\d*\.?\d+", b)
            if len(nums) >= 4:
                return [float(x) for x in nums[:4]]
    # flat X/Y/W/H keys directly on the item
    if any(k in item for k in ("X", "x", "W", "w")):
        try:
            return [float(item.get("X", item.get("x", 0))),
                    float(item.get("Y", item.get("y", 0))),
                    float(item.get("W", item.get("w", item.get("Width", 0)))),
                    float(item.get("H", item.get("h", item.get("Height", 0))))]
        except Exception:
            return None
    return None


def _split_names(v: str) -> list[str]:
    return [p.strip(" ;,") for p in re.split(r"[;/|]+", v) if p.strip(" ;,")]


def region_names(raw: dict) -> list[dict]:
    """Pull MWG/MP region names + PersonInImage as ground truth (0.99)."""
    people: list[dict] = []
    for k, v in (raw or {}).items():
        kl = k.lower()
        if "regioninfo" in kl or "personinimage" in kl or "lastkeywordxmp" in kl:
            vals = v if isinstance(v, list) else [v]
            for item in vals:
                if isinstance(item, dict):
                    name = (item.get("Name") or item.get("PersonDisplayName")
                            or item.get("PersonName") or item.get("name"))
                    box = _norm_box(item)
                    if name:
                        for nm in _split_names(str(name)):
                            if nm and not any(p["name"] == nm for p in people):
                                entry = {"name": nm, "source": "xmp_mwg_region",
                                         "confidence": 0.99}
                                if box:
                                    entry["box"] = box
                                people.append(entry)
                elif isinstance(item, str) and item.strip():
                    src = ("xmp_mwg_region" if "regioninfo" in kl
                           else "xmp_personinimage")
                    for nm in _split_names(item):
                        if nm and not any(p["name"] == nm for p in people):
                            people.append({"name": nm, "source": src,
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


def _box_overlap(a: list, b: list) -> float:
    """IoU for pixel boxes [x,y,w,h]. Region boxes (0..1) never overlap these."""
    try:
        ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
        ix = max(0, min(ax2, bx2) - max(a[0], b[0]))
        iy = max(0, min(ay2, by2) - max(a[1], b[1]))
        inter = ix * iy
        union = a[2] * a[3] + b[2] * b[3] - inter
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0


def _to01(box: list) -> list:
    """Map [x,y,w,h] into 0..1 space so pixel boxes compare with regions.

    Haar/YuNet boxes are pixels (e.g. [75,15,42,55]); MWG regions are
    normalized (e.g. [0.31,0.22,0.12,0.18]). Dividing by the box extent
    keeps small-face vs large-face geometry comparable without knowing
    the image size."""
    m = max([abs(v) for v in box[:4]] + [1e-9])
    return [v / m for v in box[:4]]


def _same_box(a: list | None, b: list | None) -> bool:
    """Match a detection box to a region box (mixed px/normalised tolerated)."""
    if not a or not b:
        return False
    try:
        an = [float(x) for x in a[:4]]
        bn = [float(x) for x in b[:4]]
    except (TypeError, ValueError):
        return False
    # both small (<2) => normalised coords, compare directly
    if max(an + bn) <= 2.0:
        return all(abs(x - y) < 0.08 for x, y in zip(an, bn))
    # mixed/large scales: compare in scale-free 0..1 space + IoU fallback
    sa, sb = _to01(an), _to01(bn)
    if all(abs(x - y) < 0.08 for x, y in zip(sa, sb)):
        return True
    return _box_overlap(sa, sb) > 0.15


_NON_NAME_WORDS = {
    "new", "folder", "image", "images", "img", "photo", "photos", "picture",
    "pictures", "copy", "screenshot", "screen", "scan", "document", "untitled",
    "unnamed", "final", "edit", "version", "download", "downloads", "dsc", "dcim",
}


def _looks_like_person_name(s: str) -> bool:
    """2-4 alphabetic words, no digits, no 'New Folder'-style words."""
    words = [w for w in str(s).split() if w]
    if not (2 <= len(words) <= 4):
        return False
    if not all(w.isalpha() and len(w) > 1 for w in words):
        return False
    return not any(w.lower() in _NON_NAME_WORDS for w in words)


def _path_name_hints(path: str | None, library: dict) -> tuple[str | None, str | None]:
    """(filename_name, folder_name) candidates — folder only if enrolled.

    'faces/sanjay agarwal.png' -> ('Sanjay Agarwal', None).
    Folder names are only trusted when they exactly match an enrolled name
    (folders are usually events: 'New folder' must never become a person)."""
    from . import faces as facemod
    if not path:
        return None, None
    norm = str(path).replace("\\\\", "/").replace("\\", "/")
    parts = [p for p in norm.split("/") if p]
    if not parts:
        return None, None
    file_name = None
    cand = facemod.clean_name(parts[-1])
    if _looks_like_person_name(cand):
        file_name = cand
    folder_name = None
    if len(parts) >= 2:
        seg = facemod.clean_name(parts[-2])
        lib_ci = {str(k).lower(): str(k) for k in (library or {})}
        if seg.lower() in lib_ci:
            folder_name = lib_ci[seg.lower()]
    return file_name, folder_name


def persons_for_faces(detections: list[dict], raw: dict, library: dict,
                      threshold: float = 0.60,
                      people_hints: list | None = None,
                      hint_conf: float = 0.5,
                      path: str | None = None) -> tuple[list[dict], int]:
    """Tag EVERY detected face. Returns (persons, unnamed_count).

    detections: [{box, embedding?}] — one entry per face in the image.
    Priority per face: region name (0.99) > local match (>=threshold) >
    filename/folder name (0.45-0.60) > vision hint (<=0.5) >
    "Unknown Person <n>" (0.2, asks for a name in _Review).
    Faces never leave the machine: embeddings compared locally only.
    """
    from . import faces as facemod  # local import: keeps people.py dependency-light
    _match = getattr(facemod, "match_face", None) or match_local
    regions = region_names(raw or {})
    hints = [str(h).split(" (")[0].strip() for h in (people_hints or []) if str(h).strip()]
    persons: list[dict] = []
    unnamed = 0
    used_regions: set[int] = set()
    used_hints: set[int] = set()
    file_hint, folder_hint = _path_name_hints(path, library)
    lib_ci = {str(k).lower(): str(k) for k in (library or {})}
    path_hint_used = False

    for i, det in enumerate(detections or []):
        box = det.get("box")
        # 1) human-labelled region overlapping this face: ground truth
        hit = None
        for ri, r in enumerate(regions):
            if ri not in used_regions and _same_box(box, r.get("box")):
                hit = (ri, r)
                break
        if hit is None and regions:
            # regions without boxes: assign in order to still-unclaimed faces
            for ri, r in enumerate(regions):
                if ri not in used_regions and not r.get("box"):
                    hit = (ri, r)
                    break
        if hit is not None:
            ri, r = hit
            used_regions.add(ri)
            entry = {"name": r["name"], "source": r.get("source", "xmp_mwg_region"),
                     "confidence": 0.99}
            if box:
                entry["box"] = box
            persons.append(entry)
            # seed the confirmed name into the library for future local matches
            try:
                emb = det.get("embedding")
                if emb and r["name"] not in library:
                    library[r["name"]] = list(emb)
            except Exception:
                pass
            continue
        # 2) local face-match above threshold (faces stay on this machine)
        m = _match(det.get("embedding"), library, threshold=threshold)
        if m:
            entry = dict(m)
            if box:
                entry["box"] = box
            persons.append(entry)
            continue
        # 2b) filename/folder name — "sanjay agarwal.png" => Sanjay Agarwal.
        # Applied to the first unmatched face only (one file, one subject).
        if not path_hint_used and (file_hint or folder_hint):
            path_hint_used = True
            if file_hint and file_hint.lower() in lib_ci:
                entry = {"name": lib_ci[file_hint.lower()],
                         "source": "face_filename", "confidence": 0.60}
            elif file_hint:
                entry = {"name": file_hint,
                         "source": "face_filename", "confidence": 0.45}
            else:
                entry = {"name": folder_hint,
                         "source": "face_folder", "confidence": 0.55}
            if box:
                entry["box"] = box
            persons.append(entry)
            continue
        # 3) vision hint (never auto-filed alone — low confidence, no box claim)
        if hints:
            hi = next((h for h in range(len(hints)) if h not in used_hints), None)
            if hi is not None:
                used_hints.add(hi)
                entry = {"name": hints[hi], "source": "vision_hint_low",
                         "confidence": min(hint_conf, 0.5)}
                if box:
                    entry["box"] = box
                persons.append(entry)
                continue
        # 4) unnamed placeholder — batched "name once" question via _Review
        unnamed += 1
        entry = {"name": f"Unknown Person {unnamed}", "source": "face_unmatched",
                 "confidence": 0.2}
        if box:
            entry["box"] = box
        persons.append(entry)
    return persons, unnamed

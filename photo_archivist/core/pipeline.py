"""Per-file pipeline orchestrator (S4): detect->fingerprint->metadata->text->vision->people->place->embed->write->decide."""
from __future__ import annotations
import os
import uuid
from .detect import detect
from .fingerprint import sha256_of, phash_of
from .metadata import exif as exifmod
from .metadata import docs as docsmod
from .metadata import video as videomod
from .metadata import fs as fsmod
from .metadata import reconcile as rec
from .metadata import place as placemod
from . import text as textmod
from . import vision as visionmod
from . import people as peoplemod
from . import place as placelib
from . import embed as embedmod
from . import pii as piimod
from . import faces as facesmod


def process_file(path: str, cfg: dict, vision_backend=None) -> dict:
    det = detect(path)
    sha = sha256_of(path)
    ph = phash_of(path) if det.type == "image" else None
    # --- metadata (never skipped) ---
    raw: dict = {}
    try:
        if det.type in ("image", "video"):
            raw = exifmod.dump(path)
    except Exception as e:
        raw = {"_exif_error": str(e)}
    fstat = fsmod.fs_stat(path)
    xa = fsmod.xattrs(path)
    takeout = fsmod.takeout_json(path)
    xmp = fsmod.xmp_sidecar(path)
    fname = fsmod.parse_filename(os.path.basename(path))
    segs = fsmod.path_segments(path)
    docmeta: dict = {}
    probe: dict = {}
    if det.type == "document":
        docmeta = docsmod.document_metadata(path, det.mime)
    if det.type == "video":
        probe = videomod.ffprobe(path)
    # --- reconcile date/place ---
    date = rec.reconcile_date(raw, fname, takeout, docmeta, fstat,
                              tz_default=cfg.get("timezone_default", "Asia/Kolkata"))
    pl = placemod.reconcile_place(raw, segs, takeout)
    if pl.get("needs_geocode") and pl.get("gps"):
        g = placelib.reverse_geocode(
            pl["gps"][0], pl["gps"][1],
            db_path=(cfg.get("geocode") or {}).get("offline_db"))
        pl["value"] = g["value"]
        pl["source"] = g["source"]
    flags = placemod.sanity_flags(raw, date)
    # --- text ---
    ocr_text, caption_extra = "", ""
    if det.mime == "application/pdf":
        t, has_layer = textmod.pdf_text(path)
        ocr_text = t
        if not has_layer:
            ocr_text = textmod.ocr_image(path) if det.type == "document" else t
    elif det.type == "document":
        ocr_text = textmod.ooxml_text(path)
    elif det.type == "image":
        # visible signage: OCR images too (cheap, local)
        try:
            ocr_text = textmod.ocr_image(path)
        except Exception:
            ocr_text = ""
    # --- event/tags from keywords + path (built first: the LLM prime uses them) ---
    tags: list[str] = []
    for k, v in raw.items():
        kl = k.lower()
        if any(s in kl for s in ("keyword", "subject", "hierarchicalsubject")):
            if isinstance(v, list):
                tags.extend(str(x) for x in v)
            elif v:
                tags.append(str(v))
    tags.extend(segs[-3:])
    event_val = tags[0] if tags else None
    # --- vision (primed with text-first context; mock | local | llm backend) ---
    vb = vision_backend or visionmod.get_backend(
        (cfg.get("vision") or {}).get("backend", "mock"), cfg)
    prime = {
        "place": pl.get("value"),
        "event_hint": " / ".join(segs[-2:]) if segs else "",
        "file_name": os.path.basename(path),
        "file_type": det.type, "mime": det.mime,
        "path_segments": segs,
        "ocr_text": (ocr_text or "")[:2000],
        "metadata": {"taken_at": date.get("taken_at"),
                     "taken_at_source": date.get("source"),
                     "place": pl.get("value"), "tags": sorted(set(tags))[:20],
                     "camera": (raw.get("EXIF:Make") or raw.get("Make")),
                     "photographer": raw.get("IPTC:By-line") or raw.get("XMP-dc:creator")},
    }
    vr = vb.describe(path, prime)
    if vr.tags:                       # LLM-provided tags enrich searchability
        tags.extend(str(t) for t in vr.tags)
    # --- people: XMP region names + LLM text hints + local face matches ---
    persons = peoplemod.region_names(raw)
    for h in vr.people_hints or []:
        name = str(h).split(" (")[0].strip()
        if name and all(name != p["name"] for p in persons):
            persons.append({"name": name, "source": "llm_text_extract",
                            "confidence": min(0.9, vr.confidences.get("llm", 0.5))})
    faces_cfg = cfg.get("faces") or {}
    if vr.face_embeddings:
        face_lib = peoplemod.load_face_library(faces_cfg.get("library", "faces_library.json"))
        if face_lib:
            thr = float(faces_cfg.get("match_threshold", 0.60))
            for emb in vr.face_embeddings:
                m = peoplemod.match_local(emb, face_lib, thr)
                if m and all(m["name"] != p["name"] for p in persons):
                    persons.append(m)
    # --- faces: OpenCV Haar detection (boxes only; identity matching dormant) ---
    face_boxes = facesmod.detect_faces(path) if det.type == "image" else []
    # --- embed (local hash only — org policy forbids model downloads) ---
    emb_backend = (cfg.get("embeddings") or {}).get("backend", "hash")
    txt_vec = embedmod.text_vector(
        " ".join([vr.caption, " ".join(tags), ocr_text, " ".join(segs)]),
        backend=emb_backend)
    # --- record (S7) ---
    pii_flags = piimod.scan_text(ocr_text, vr.caption, vr.visible_text)
    pii_flags += [f"llm:{f}" for f in (vr.pii_flags or [])]
    if not raw:
        meta_status = "stripped"
    elif pl.get("gps") or date.get("confidence", 0) >= 0.8:
        meta_status = "complete"
    else:
        meta_status = "partial"
    rec_out = {
        "file_id": uuid.uuid4().hex[:12], "sha256": sha, "phash": ph,
        "source_path": path, "organised_path": None,
        "type": det.type, "mime": det.mime,
        "taken_at": date.get("taken_at"), "taken_at_source": date.get("source"),
        "taken_at_confidence": date.get("confidence"),
        "timezone_source": date.get("timezone_source"),
        "place": {k: v for k, v in pl.items() if k != "needs_geocode"},
        "people": persons, "face_boxes": face_boxes,
        "event": {"value": event_val, "source": "iptc_keywords|path" if event_val else None,
                  "confidence": 0.7 if event_val else 0.0},
        "caption": vr.caption, "tags": sorted(set(tags)),
        "ocr_text": ocr_text[:5000],
        "camera": {"make": raw.get("EXIF:Make") or raw.get("Make"),
                   "model": raw.get("EXIF:Model") or raw.get("Model"),
                   "serial": raw.get("EXIF:SerialNumber") or raw.get("SerialNumber"),
                   "lens": raw.get("EXIF:LensModel") or raw.get("LensModel")},
        "photographer": raw.get("IPTC:By-line") or raw.get("XMP-dc:creator"),
        "rating": raw.get("XMP-xmp:Rating") or raw.get("Rating"),
        "document": docmeta, "duplicates": [], "cluster_id": None,
        "pii_flags": pii_flags + flags, "metadata_status": meta_status,
        "raw_metadata": {"exif": raw, "fs": fstat, "xattr": bool(xa),
                         "takeout": bool(takeout), "xmp_sidecar": bool(xmp),
                         "filename": fname, "probe": bool(probe)},
        "vectors": {"image": vr.image_vec, "text": txt_vec},
        "_segs": segs,
    }
    return rec_out

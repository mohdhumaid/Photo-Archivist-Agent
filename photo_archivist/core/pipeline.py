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
    human: dict = {}
    try:
        if det.type in ("image", "video"):
            human = exifmod.dump_human(path)
            for k, v in human.items():
                raw.setdefault(k, v)
    except Exception:
        pass
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
    meta_place_conf = float(pl.get("confidence") or 0)
    flags = placemod.sanity_flags(raw, date)
    # --- text ---
    ocr_text = ""
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
    # --- event/tags: human keywords first (§3.4 trust order), then path.
    # Never take the vision caption as the event.
    def _kw_split(v) -> list[str]:
        if isinstance(v, list):
            out: list[str] = []
            for x in v:
                out.extend(_kw_split(x))
            return out
        s = str(v or "").strip()
        if not s:
            return []
        # hierarchical 'Events|Branch Launch|Indiranagar' -> last leaf first
        parts = [p.strip() for chunk in s.split(";") for p in str(chunk).split("|")]
        return [p for p in parts if p]
    kw_leaves: list[str] = []
    kw_full: list[str] = []
    for k, v in raw.items():
        kl = k.lower()
        if any(s in kl for s in ("hierarchicalsubject",)):
            for leaf in _kw_split(v):
                if leaf not in kw_full:
                    kw_full.append(leaf)
            leaves = [_kw_split(v)[-1]] if _kw_split(v) else []
            for lf in leaves:
                if lf not in kw_leaves:
                    kw_leaves.append(lf)
        elif any(s in kl for s in ("keyword", "subject", "category")):
            for leaf in _kw_split(v):
                if leaf not in kw_full:
                    kw_full.append(leaf)
            if str(v).strip() and str(v).strip() not in kw_leaves:
                pass
    tags: list[str] = []
    tags.extend(kw_full)
    # takeout description / album membership often names the event
    if isinstance(takeout, dict):
        for tk in ("description", "title"):
            tv = takeout.get(tk)
            if isinstance(tv, str) and tv.strip() and tv.strip() not in tags:
                tags.append(tv.strip()[:120])
    # document text fallback: sanction letters etc name their own event
    doctitle = ""
    if isinstance(docmeta, dict):
        info = docmeta.get("info") if isinstance(docmeta.get("info"), dict) else {}
        doctitle = str((info or {}).get("Title") or
                       (docmeta.get("docProps/core.xml") or {}).get("title") or "").strip()
    if doctitle and doctitle not in tags:
        tags.append(doctitle[:120])
    tags.extend([s for s in segs[-3:] if s not in tags
                 and "sample_data" not in s and s not in (".", "/")])
    # never let the drive-root / generic crawl folder become a tag or event
    tags = [t for t in tags if t not in ("sample_data", "data", "source", "src")]
    event_val, event_src, event_conf = None, None, 0.0
    if kw_leaves:
        event_val, event_src, event_conf = kw_leaves[0], "iptc_keywords", 0.93
    elif kw_full:
        event_val, event_src, event_conf = kw_full[0], "iptc_keywords", 0.85
    elif doctitle:
        event_val, event_src, event_conf = doctitle[:120], "document_title", 0.8
    elif segs:
        event_val, event_src, event_conf = segs[-1], "path", 0.5
    # XMP sidecar tags live outside exiftool's dump — parse names cheaply
    if xmp:
        import re as _re
        for m in _re.findall(r"(?:dc:subject|rdf:li|PersonInImage|hierarchicalSubject)"
                             r"[^<>]*>([^<>]{2,80})<", xmp):
            mm = m.strip()
            if mm and mm not in tags:
                tags.append(mm)
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
        for t in vr.tags:
            if str(t) not in tags:
                tags.append(str(t))
    # banners/visible text are the most reliable corporate-event signal
    if vr.visible_text and vr.visible_text not in tags:
        tags.append(vr.visible_text[:120])
    # LLM location fallback: ONLY when metadata gave nothing concrete AND the
    # guess names a specific place with evidence (generic words already nulled
    # by vllm.scrub_location_guess; mock backend has no location_guess at all).
    # Metadata trust order (§3.4) always wins on any conflict. A bare tmp/
    # crawl path (pytest tmp_path etc) is not a real place — it yields too.
    loc = getattr(vr, "location_guess", None) or {}
    # A bare tmp/pytest crawl path is not a real place — the LLM may fill it.
    # Any other path value (real folder names, conf 0.4) keeps its priority.
    _pv = str(pl.get("value") or "")
    _junk_path = (pl.get("source") == "path"
                  and any(s in _pv for s in ("tmp", "tmpp", "pytest-")))
    if ((not pl.get("value")) or _junk_path) and loc.get("value") \
            and float(loc.get("confidence") or 0) >= 0.5:
        pl = {"value": loc["value"],
              "source": f"vision_location:{vr.backend}",
              "gps": None, "accuracy_m": None,
              "confidence": min(float(loc.get("confidence") or 0.5), 0.7),
              "evidence": loc.get("evidence") or ""}
    elif loc.get("rejected"):
        flags.append(f"vision_location_rejected:{loc['rejected']}")
    if vr.event_type and not event_val:
        event_val, event_src, event_conf = vr.event_type, "vision_event", 0.5
    elif vr.event_type and event_src == "path":
        # vision corroborates a path-only guess: keep value, note corroboration
        event_src = "path+vision"
    # --- people: tag EVERY face (region ground truth > local match > hint) ---
    # --- people: tag EVERY face (region ground truth > local match > hint) ---
    faces_cfg = cfg.get("faces") or {}
    face_lib_path = faces_cfg.get("library", "faces_library.json")
    face_lib = peoplemod.load_face_library(face_lib_path)
    lib_before = set(face_lib.keys())
    thr = float(faces_cfg.get("match_threshold", 0.40))
    # One detection entry per face: vision boxes + per-face embeddings when given.
    detections: list[dict] = []
    for _i, _b in enumerate(vr.face_boxes or []):
        _emb = None
        try:
            if vr.face_embeddings and _i < len(vr.face_embeddings):
                _emb = vr.face_embeddings[_i]
        except Exception:
            _emb = None
        if _b:
            detections.append({"box": _b, "embedding": _emb})
    # --- faces: OpenCV YuNet / Haar detection (local only, every face) ---
    face_boxes = facesmod.detect_faces(path, cfg) if det.type == "image" else []
    if det.type == "image" and not detections:
        for fb in face_boxes:
            _box = fb.get("box") if isinstance(fb, dict) else fb
            if _box:
                detections.append({"box": _box, "embedding": None,
                                   "raw_face": fb.get("raw_face") if isinstance(fb, dict) else None})
    # Vision-boxed faces: attach YuNet 5-point landmarks by overlap so the
    # embedding uses alignCrop (aligned space == enrollment space; a bare
    # box crop scores ~0.28 cosine vs ~0.87 aligned and never matches).
    if detections and face_boxes:
        for _d in detections:
            if not _d.get("raw_face"):
                for fb in face_boxes:
                    if isinstance(fb, dict) and peoplemod._same_box(_d.get("box"), fb.get("box")):
                        _d["raw_face"] = fb.get("raw_face")
                        break
    # Fill missing per-face embeddings locally (faces never leave the machine).
    if det.type == "image":
        for _d in detections:
            if not _d.get("embedding"):
                try:
                    _e = facesmod.embed_face(path, {"box": _d.get("box"),
                                                    "raw_face": _d.get("raw_face")}, cfg)
                    if _e is None:
                        _e = facesmod.crop_vector(path, _d.get("box"))
                    _d["embedding"] = _e
                except Exception:
                    pass
    persons, _unnamed_n = peoplemod.persons_for_faces(
        detections, raw, face_lib, threshold=thr,
        people_hints=list(vr.people_hints or []), path=path)
    # Ground-truth region names with no detection box still count - never drop
    # a human label even when the detector missed that face.
    try:
        _have = {p.get("name") for p in persons}
        for _r in peoplemod.region_names(raw or {}):
            if _r.get("name") and _r["name"] not in _have:
                persons.append(_r)
                _have.add(_r["name"])
    except Exception:
        pass
    # Persist newly seeded region names so the same face matches next scan.
    try:
        if set(face_lib.keys()) - lib_before:
            import json as _json
            with open(face_lib_path, "w") as _f:
                _json.dump(face_lib, _f)
    except Exception:
        pass
    if det.type == "image" and not face_boxes:
        face_boxes = [_d.get("box") for _d in detections if _d.get("box")]
    # --- embed (local hash only — org policy forbids model downloads) ---
    emb_backend = (cfg.get("embeddings") or {}).get("backend", "hash")
    txt_vec = embedmod.text_vector(
        " ".join([vr.caption, " ".join(tags), ocr_text, " ".join(segs)]),
        backend=emb_backend)
    # --- record (S7): every field carries source+confidence; unknown -> null ---
    pii_flags = piimod.scan_text(ocr_text, vr.caption, vr.visible_text)
    pii_flags += [f"llm:{f}" for f in (vr.pii_flags or [])]
    if not raw and not docmeta and not takeout:
        meta_status = "stripped"
    elif pl.get("gps") or date.get("confidence", 0) >= 0.8:
        meta_status = "complete"
    else:
        meta_status = "partial"
    # caption: human-written text wins over any model caption
    human_caption = (
        raw.get("IPTC:Caption-Abstract") or raw.get("XMP-dc:description")
        or raw.get("XMP-photoshop:Headline") or raw.get("EXIF:ImageDescription")
        or raw.get("EXIF:UserComment") or (takeout or {}).get("description") or "")
    if isinstance(human_caption, list):
        human_caption = " ".join(str(x) for x in human_caption)
    caption = str(human_caption).strip() or vr.caption
    cap_src = ("iptc_caption" if human_caption else vr.backend or "mock")
    # photo description: background + activity + mood (+ scene), LLM only
    photo_desc_bits = [b for b in (getattr(vr, "background", ""),
                                   getattr(vr, "people_activity", ""),
                                   getattr(vr, "mood", ""),
                                   getattr(vr, "scene", "")) if b]
    photo_description = "; ".join(photo_desc_bits)[:500] or None
    photo_desc_src = vr.backend if photo_desc_bits and vr.backend != "mock" else None
    # photographer: By-line / creator / Takeout / camera owner
    photographer = (
        raw.get("IPTC:By-line") or raw.get("XMP-dc:creator")
        or raw.get("EXIF:Artist") or raw.get("EXIF:CameraOwnerName")
        or (takeout or {}).get("photographer") or None)
    if isinstance(photographer, list):
        photographer = "; ".join(str(x) for x in photographer)
    try:
        rating_v = raw.get("XMP-xmp:Rating") or raw.get("Rating")
        rating = int(rating_v) if rating_v not in (None, "") else None
    except Exception:
        rating = None
    # camera serial: any group; lens: any group
    def _any(*names):
        for nm in names:
            for k, v in raw.items():
                if k.split(":")[-1].lower() == nm.lower() and v not in (None, ""):
                    return v
        return None
    camera = {"make": _any("Make"), "model": _any("Model"),
              "serial": _any("SerialNumber", "BodySerialNumber", "CameraSerialNumber"),
              "lens": _any("LensModel", "Lens"),
              "orientation": _any("Orientation")}
    # shoot cluster: same serial + same minute (+ same folder)
    cluster_id = None
    try:
        day = str(date.get("taken_at") or "")[:16].replace(" ", "T")
        serial = str(camera.get("serial") or camera.get("model") or "noserial")
        folder = segs[-1] if segs else "root"
        if date.get("taken_at"):
            import hashlib as _hl
            cluster_id = "shoot_" + _hl.md5(
                f"{day}|{serial}|{folder}".encode()).hexdigest()[:12]
    except Exception:
        cluster_id = None
    rec_out = {
        "file_id": uuid.uuid4().hex[:12], "sha256": sha, "phash": ph,
        "source_path": path, "organised_path": None,
        "type": det.type, "mime": det.mime,
        "taken_at": date.get("taken_at"), "taken_at_source": date.get("source"),
        "taken_at_confidence": date.get("confidence"),
        "timezone_source": date.get("timezone_source"),
        "place": {k: v for k, v in pl.items() if k != "needs_geocode"},
        "people": persons, "face_boxes": face_boxes,
        "event": {"value": event_val, "source": event_src, "confidence": event_conf},
        "caption": caption, "caption_source": cap_src,
        "photo_description": photo_description,
        "photo_description_source": photo_desc_src,
        "tags": sorted(set(tags)),
        "ocr_text": ocr_text[:5000],
        "camera": camera,
        "photographer": photographer,
        "rating": rating,
        "document": docmeta, "duplicates": [], "cluster_id": cluster_id,
        "pii_flags": pii_flags + flags, "metadata_status": meta_status,
        "raw_metadata": {"exif": raw, "human_exif": human,
                         "fs": fstat, "xattr": xa,
                         "takeout": takeout, "xmp_sidecar": xmp,
                         "filename": fname, "probe": probe},
        "vectors": {"image": vr.image_vec, "text": txt_vec},
        "_segs": segs,
    }
    return rec_out

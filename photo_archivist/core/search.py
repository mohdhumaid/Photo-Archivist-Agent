"""Search (S6): parse -> exact metadata filter -> hybrid -> verify -> reason+provenance."""
from __future__ import annotations
import re
import sqlite3

# Fields indexed in files_fts (keep in sync with index.SCHEMA).
_TEXT_FIELDS = ("caption", "photo_description", "tags", "ocr_text", "source_path",
                "place_value", "event_value", "organised_path")

_MONTHS = {"jan": "01", "feb": "02", "mar": "03", "apr": "04",
           "may": "05", "jun": "06", "jul": "07", "aug": "08",
           "sep": "09", "oct": "10", "nov": "11", "dec": "12"}
_MONTHS_REV = {v: k for k, v in _MONTHS.items()}


def _folder_text(r: dict) -> str:
    """organised_path normalised for date/place searching."""
    op = str(r.get("organised_path") or "")
    return (op.replace("/", " ").replace(chr(92), " ")
              + " " + op.replace("/", "").replace(chr(92), "")).lower()


def _blob(r: dict) -> str:
    """Lowercased text haystack used for substring matching / reasons."""
    bits = [str(r.get(k) or "") for k in _TEXT_FIELDS]
    # folder tokens work twice: '2025/2025-07/...' as pieces AND spaceless,
    # so '202507'/'20250714' queries hit year-month paths too.
    op = str(r.get("organised_path") or "")
    bits.append(op.replace("/", " ").replace(chr(92), " "))
    bits.append(op.replace("/", "").replace(chr(92), ""))
    return " ".join(bits).lower()


def _term_aliases(term: str) -> set:
    """Alternate spellings for one query term (month bridging only).

    Folder paths store numbers ('2025-07'); users type words ('july').
    A term counts as matched when it OR any alias appears in the row text.
    """
    w = (term or "").lower()
    if w in _MONTHS:                              # 'july' -> '07', 'jul'
        return {_MONTHS[w], w[:3]}
    for name, num in _MONTHS.items():             # 'jul' -> '07'
        if w == name:
            return {num}
    return set()


def parse_query(q: str, role_map: dict | None = None) -> dict:
    f: dict = {"terms": [], "person": None, "place": None, "year": None,
           "year_month": None, "month": None, "type": None}
    m = re.search(r"\b((?:19|20)\d{2})(?:[-/]?(0?[1-9]|1[0-2]))?\b", q or "")
    if m:
        f["year"] = m.group(1)
        if m.group(2):
            f["year_month"] = f"{f['year']}-{m.group(2).zfill(2)}"
    low0 = (q or "").lower()
    for name, num in _MONTHS.items():
        if re.search(rf"\b{name}", low0) or re.search(rf"\b{name[:3]}", low0):
            f["month"] = num
            break
    if f["year"] and f["month"] and not f["year_month"]:
        f["year_month"] = f"{f['year']}-{f['month']}"   # 'july 2025' -> 2025-07
    low = q.lower()
    # Only treat as type filter when the word stands alone-ish; 'letter' is
    # usually content ("sanction letter"), not a type request.
    type_words = {"image": "image", "photo": "image", "photos": "image",
                  "pdf": "document", "document": "document", "documents": "document",
                  "video": "video", "videos": "video"}
    for t, mapped in type_words.items():
        if re.search(rf"(^|[\s\"'])({t})($|[\s\"'])", low):
            # 'photo ...' etc: but if query also has other content terms keep filter;
            # single-word queries like 'videos' clearly mean type.
            f["type"] = mapped
            break
    # whole-word role match only ("MD letter" yes, "BM" inside "bmw" no)
    for role, name in (role_map or {}).items():
        if role and name and re.search(rf"\b{re.escape(str(role).lower())}\b", low):
            f["person"] = name
            break
    # quoted person heuristic
    m2 = re.search(r'"([^"]+)"', q)
    if m2 and not f["person"]:
        f["person"] = m2.group(1)
    # strip surrounding quotes/punct so '"Anita Rao"' -> ['Anita', 'Rao']
    f["terms"] = [w.strip("\"'") for w in re.split(r"\s+", q)
                  if len(w.strip("\"',")) > 2][:10]
    return f


def search(db_path: str, query: str, role_map: dict | None = None, limit: int = 20,
           people_map: dict | None = None) -> list[dict]:
    filt = parse_query(query, role_map)
    # Short queries ("MR", "MD") are dropped by the len>2 term rule — fall back
    # to the raw query as one substring term so we never return everything.
    if not filt["terms"] and query.strip() and not filt["person"]:
        filt["terms"] = [query.strip()]
    c = sqlite3.connect(db_path)
    c.row_factory = sqlite3.Row
    sql = "SELECT * FROM files WHERE 1=1"
    args: list = []
    # NOTE: no SQL-level year filter — taken_at often holds template defaults
    # while the true year lives in OCR/text. Year handled broadly below.
    if filt["type"]:
        sql += " AND type=?"
        args.append(filt["type"])
    rows = [dict(r) for r in c.execute(sql, args).fetchall()]
    # Year: match taken_at OR any text field (OCR often carries the true year
    # while container dates hold template defaults).
    if filt["year"]:
        y = filt["year"]
        rows = [r for r in rows
                if y in (r["taken_at"] or "")
                or y in (r["ocr_text"] or "") or y in (r["caption"] or "")
                or y in (r["tags"] or "") or y in (r["source_path"] or "")
                or y in _folder_text(r)]
    # month-qualified queries: broaden the evidence to taken_at, folders and
    # free text (YYYY-MM/YYYYMM in paths: '202507' hits '2025/2025-07/...').
    if filt["year_month"] or filt["month"]:
        ym = filt["year_month"]
        needles = {ym, ym.replace("-", "")} if ym else set()
        if filt["month"]:
            needles.add(filt["month"])                   # numeric '07'
            needles.add(_MONTHS_REV[filt["month"]])      # short 'jul'
        rows = [r for r in rows
                if any(n in (r["taken_at"] or "").replace("-", "").replace(":", "")
                       or n in _folder_text(r) or n in _blob(r) for n in needles)]
    # FTS: prefix match each term ("Akhil" matches "AkhilVerma"). FTS returns
    # rowids; map them back to file_id (files.rowid == files_fts.rowid).
    try:
        fts_terms = " OR ".join(f'"{t}"*' for t in filt["terms"]) or f'"{query}"*'
        hit_rows = {r[0] for r in c.execute(
            "SELECT rowid FROM files_fts WHERE files_fts MATCH ?",
            (fts_terms,)).fetchall()}
    except Exception:
        hit_rows = set()
    hits: set = set()
    if hit_rows:
        hits = {r[0] for r in c.execute(
            "SELECT file_id FROM files WHERE rowid IN (%s)"
            % ",".join("?" * len(hit_rows)), tuple(hit_rows)).fetchall()}
    # person filter via people table
    if filt["person"]:
        prow = c.execute("SELECT file_id FROM people WHERE name LIKE ?",
                         (f"%{filt['person']}%",)).fetchall()
        pset = {r[0] for r in prow}
        rows = [r for r in rows if r["file_id"] in pset]
    # --- known people named in the query (whole-token match, e.g. "sanjay"
    #     picking out "Sanjay Agarwal"); drives person-first ranking ---
    import re as _re
    q_tokens = set(_re.findall(r"[a-z0-9]+", (query or "").lower()))
    query_persons: set[str] = set()
    if filt.get("person"):
        query_persons.add(str(filt["person"]).lower())
    try:
        names = [str(nm) for (nm,) in
                 c.execute("SELECT DISTINCT name FROM people").fetchall()
                 if nm and not str(nm).lower().startswith("unknown")]
        # token -> names containing it; only UNIQUE tokens may identify a
        # person ('jain' -> Yogesh Jain, but 'yogesh' is ambiguous between
        # Yogesh Jain / Yogesh Soni and must not boost either).
        tok_names: dict[str, set[str]] = {}
        for nm in names:
            for t in _re.findall(r"[a-z0-9]+", nm.lower()):
                tok_names.setdefault(t, set()).add(nm.lower())
        for nm in names:
            ntoks = _re.findall(r"[a-z0-9]+", nm.lower())
            if any(t in q_tokens and len(tok_names.get(t, ())) == 1
                   for t in ntoks):
                query_persons.add(nm.lower())
    except Exception:
        pass
    people_by_file: dict[str, list[str]] = {}
    try:
        for fid, nm in c.execute("SELECT file_id, name FROM people").fetchall():
            if nm:
                people_by_file.setdefault(fid, []).append(str(nm))
    except Exception:
        pass

    # --- relevance: tier-1 (ALL terms / full phrase / named person) wins;
    #     otherwise degrade to >=0.5 coverage, ranked by semantic score. ---
    from . import embed as embedmod
    from .index import _vec_from_blob
    qvec = embedmod.text_vector(" ".join(filt["terms"]) or (query or ""))
    norm_q = " ".join((query or "").lower().split())
    scored: list[dict] = []
    for r in rows:
        blob = " ".join(_blob(r).split())
        matched = [t for t in filt["terms"]
                   if t.lower() in blob
                   or any(a in blob for a in _term_aliases(t))]
        coverage = (len(matched) / len(filt["terms"])) if filt["terms"] else 1.0
        phrase = bool(norm_q) and norm_q in blob
        fts = r["file_id"] in hits
        person_hit = any(p.lower() in query_persons
                         for p in people_by_file.get(r["file_id"], []))
        vec = _vec_from_blob(r["txt_vec"])
        sem = embedmod.cosine(qvec, vec) if vec else 0.0
        score = (3.0 * phrase) + (2.5 * person_hit) + (1.0 * coverage) \
                + (0.6 * sem) + (0.3 * fts) + (0.05 * (r["taken_at_conf"] or 0))
        scored.append({"r": r, "matched": matched, "score": score,
                       "full": coverage >= 1.0 or phrase or person_hit,
                       "coverage": coverage, "phrase": phrase,
                       "person_hit": person_hit})
    if filt["terms"]:
        tier1 = [s for s in scored if s["full"]]
        if tier1:
            chosen = tier1          # precision: drop partial-keyword noise
        else:
            floor = 0.5 if len(filt["terms"]) >= 2 else 1.0
            # rows that survived a year/month filter carry date evidence even
            # when the digits aren't repeated in the text blob ('2026' files).
            date_ok = bool(filt.get("year") or filt.get("year_month") or filt.get("month"))
            chosen = [s for s in scored if s["coverage"] >= floor
                      or s["phrase"] or s["person_hit"] or date_ok]
    else:
        chosen = scored
    chosen.sort(key=lambda s: -s["score"])
    out = []
    for s in chosen[:limit]:
        r = s["r"]
        ppl = c.execute("SELECT name,source,conf FROM people WHERE file_id=?",
                        (r["file_id"],)).fetchall()
        reason_bits = []
        if r["place_value"]:
            reason_bits.append(f"{r['place_value']} (from {r['place_source']})")
        if ppl:
            # annotate matched names with their library titles (people: map)
            reason_bits.append(", ".join(
                p[0] + f" [{p[1]}]"
                + (f" — {(people_map or {}).get(p[0])}" if (people_map or {}).get(p[0]) else "")
                for p in ppl))
        if r["taken_at"]:
            reason_bits.append(f"{r['taken_at']} (from {r['taken_at_source']})")
        if s["matched"]:
            reason_bits.append("text match: " + ", ".join(s["matched"]))
        if s["phrase"]:
            reason_bits.append("exact phrase")
        if s["person_hit"]:
            reason_bits.append("named person match")
        out.append({"file_id": r["file_id"], "source_path": r["source_path"],
                    "organised_path": r["organised_path"], "caption": r["caption"],
                    "reason": " · ".join(reason_bits) or "tag/vector match",
                    "in_fts": r["file_id"] in hits,
                    "score": round(s["score"], 3)})
    c.close()
    return out

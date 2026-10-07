"""Search (S6): parse -> exact metadata filter -> hybrid -> verify -> reason+provenance."""
from __future__ import annotations
import re
import sqlite3

# Fields indexed in files_fts (keep in sync with index.SCHEMA).
_TEXT_FIELDS = ("caption", "photo_description", "tags", "ocr_text", "source_path",
                "place_value", "event_value")


def _blob(r: dict) -> str:
    """Lowercased text haystack used for substring matching / reasons."""
    return " ".join(str(r.get(k) or "") for k in _TEXT_FIELDS).lower()


def parse_query(q: str, role_map: dict | None = None) -> dict:
    f: dict = {"terms": [], "person": None, "place": None, "year": None, "type": None}
    m = re.search(r"\b(19|20)\d{2}\b", q)
    if m:
        f["year"] = m.group(0)
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
                or y in (r["tags"] or "") or y in (r["source_path"] or "")]
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
    # content filter: keep only rows that actually match the query terms
    # (FTS prefix hit OR case-insensitive substring across indexed text).
    if filt["terms"]:
        tl = [t.lower() for t in filt["terms"]]
        rows = [r for r in rows
                if r["file_id"] in hits or any(t in _blob(r) for t in tl)]
    # rank: FTS hit first, then confidence
    rows.sort(key=lambda r: (0 if r["file_id"] in hits else 1,
                             -((r["taken_at_conf"] or 0))))
    out = []
    for r in rows[:limit]:
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
        if filt["terms"]:
            matched = [t for t in filt["terms"] if t.lower() in _blob(r)]
            if matched:
                reason_bits.append("text match: " + ", ".join(matched))
        out.append({"file_id": r["file_id"], "source_path": r["source_path"],
                    "organised_path": r["organised_path"], "caption": r["caption"],
                    "reason": " · ".join(reason_bits) or "tag/vector match",
                    "in_fts": r["file_id"] in hits})
    c.close()
    return out

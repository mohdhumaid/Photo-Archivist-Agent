"""Step 9: Index — SQLite schema matching S7 + FTS5 (+ folder profiles)."""
from __future__ import annotations
import json
import os
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS files(
 file_id TEXT PRIMARY KEY, sha256 TEXT UNIQUE, phash TEXT,
 source_path TEXT, organised_path TEXT, type TEXT, mime TEXT,
 taken_at TEXT, taken_at_source TEXT, taken_at_conf REAL, tz_source TEXT,
 place_value TEXT, place_source TEXT, gps_lat REAL, gps_lon REAL,
 accuracy_m REAL, place_conf REAL,
 event_value TEXT, event_source TEXT, event_conf REAL,
 caption TEXT, photo_description TEXT, tags TEXT, ocr_text TEXT,
 camera TEXT, photographer TEXT,
 rating INTEGER, document TEXT, duplicates TEXT, cluster_id TEXT,
 pii_flags TEXT, metadata_status TEXT, raw_metadata TEXT,
 img_vec BLOB, txt_vec BLOB);
CREATE TABLE IF NOT EXISTS people(
 file_id TEXT, name TEXT, source TEXT, conf REAL, box TEXT);
CREATE TABLE IF NOT EXISTS folders(
 name TEXT PRIMARY KEY, centroid BLOB, tags TEXT, people TEXT,
 place TEXT, date_from TEXT, date_to TEXT, pattern TEXT);
CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5(
 caption, photo_description, tags, ocr_text, source_path, place_value, event_value);
"""


def connect(db_path: str) -> sqlite3.Connection:
    c = sqlite3.connect(db_path)
    c.executescript(SCHEMA)
    return c


def _vec_blob(v) -> bytes | None:
    import struct
    if not v:
        return None
    try:
        return struct.pack(f"{len(v)}f", *[float(x) for x in v])
    except Exception:
        return None


def _j(v) -> str:
    try:
        return json.dumps(v, default=str)
    except Exception:
        return json.dumps(str(v))


def upsert_file(c: sqlite3.Connection, rec: dict) -> None:
    c.execute(
        """INSERT OR REPLACE INTO files(file_id,sha256,phash,source_path,organised_path,
        type,mime,taken_at,taken_at_source,taken_at_conf,tz_source,place_value,place_source,
        gps_lat,gps_lon,accuracy_m,place_conf,event_value,event_source,event_conf,
        caption,photo_description,tags,ocr_text,camera,photographer,rating,document,duplicates,cluster_id,
        pii_flags,metadata_status,raw_metadata,img_vec,txt_vec)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (rec.get("file_id"), rec.get("sha256"), rec.get("phash"), rec.get("source_path"),
         rec.get("organised_path"), rec.get("type"), rec.get("mime"), rec.get("taken_at"),
         rec.get("taken_at_source"), rec.get("taken_at_confidence"),
         rec.get("timezone_source"), (rec.get("place") or {}).get("value"),
         (rec.get("place") or {}).get("source"),
         ((rec.get("place") or {}).get("gps") or [None, None])[0] if (rec.get("place") or {}).get("gps") else None,
         ((rec.get("place") or {}).get("gps") or [None, None])[1] if (rec.get("place") or {}).get("gps") else None,
         (rec.get("place") or {}).get("accuracy_m"), (rec.get("place") or {}).get("confidence"),
         (rec.get("event") or {}).get("value"), (rec.get("event") or {}).get("source"),
         (rec.get("event") or {}).get("confidence"), rec.get("caption"),
         rec.get("photo_description"),
         _j(rec.get("tags", [])), rec.get("ocr_text"), _j(rec.get("camera", {})),
         rec.get("photographer"), rec.get("rating"), _j(rec.get("document", {})),
         _j(rec.get("duplicates", [])), rec.get("cluster_id"),
         _j(rec.get("pii_flags", [])), rec.get("metadata_status"),
         _j(rec.get("raw_metadata", {})),
         _vec_blob((rec.get("vectors") or {}).get("image")),
         _vec_blob((rec.get("vectors") or {}).get("text"))),
    )
    c.execute("DELETE FROM people WHERE file_id=?", (rec.get("file_id"),))
    for p in rec.get("people", []) or []:
        c.execute("INSERT INTO people(file_id,name,source,conf,box) VALUES(?,?,?,?,?)",
                  (rec.get("file_id"), p.get("name"), p.get("source"),
                   p.get("confidence"), json.dumps(p.get("box"))))
    # re-scan safety: drop any stale FTS row for this file's rowid first
    c.execute(
        "DELETE FROM files_fts WHERE rowid="
        "(SELECT rowid FROM files WHERE file_id=?)",
        (rec.get("file_id"),))
    c.execute(
        "INSERT INTO files_fts(rowid,caption,photo_description,tags,ocr_text,source_path,place_value,event_value)"
        " VALUES((SELECT rowid FROM files WHERE file_id=?),?,?,?,?,?,?,?)",
        (rec.get("file_id"), rec.get("caption") or "", rec.get("photo_description") or "",
         " ".join(rec.get("tags", []) or []),
         rec.get("ocr_text") or "", rec.get("source_path") or "",
         (rec.get("place") or {}).get("value") or "", (rec.get("event") or {}).get("value") or ""),
    )
    c.commit()


# --- folder profiles: the learning loop behind thresholds.promote ----------

def _vec_from_blob(b: bytes | None) -> list | None:
    if not b:
        return None
    import struct
    try:
        return list(struct.unpack(f"{len(b) // 4}f", b))
    except Exception:
        return None


def profile_from_records(name: str, recs: list[dict]) -> dict:
    """Fold the records filed into one folder into a decide()-ready profile."""
    vecs = [r["vectors"]["text"] for r in recs
            if (r.get("vectors") or {}).get("text")]
    centroid: list | None = None
    if vecs:
        d = len(vecs[0])
        centroid = [sum(v[i] for v in vecs) / len(vecs) for i in range(d)]
    tags = sorted({t for r in recs for t in (r.get("tags") or [])})
    people = sorted({p["name"] for r in recs for p in (r.get("people") or [])})
    dates = sorted(r["taken_at"] for r in recs if r.get("taken_at"))
    places = [ (r.get("place") or {}).get("value") for r in recs
               if (r.get("place") or {}).get("value") ]
    return {"name": name, "centroid": centroid, "tags": tags, "people": people,
            "place": places[0] if places else None,
            "date_from": dates[0] if dates else None,
            "date_to": dates[-1] if dates else None, "pattern": None}


def upsert_folder(c: sqlite3.Connection, prof: dict) -> None:
    c.execute(
        """INSERT OR REPLACE INTO folders
           (name, centroid, tags, people, place, date_from, date_to, pattern)
           VALUES(?,?,?,?,?,?,?,?)""",
        (prof.get("name"), _vec_blob(prof.get("centroid")),
         json.dumps(prof.get("tags") or []), json.dumps(prof.get("people") or []),
         prof.get("place"), prof.get("date_from"), prof.get("date_to"),
         prof.get("pattern")))
    c.commit()


def load_folders(db_path: str) -> list[dict]:
    """All persisted folder profiles (empty list when the index doesn't exist)."""
    if not db_path or not os.path.exists(db_path):
        return []
    c = connect(db_path)
    out: list[dict] = []
    for r in c.execute(
            "SELECT name, centroid, tags, people, place, date_from, date_to,"
            " pattern FROM folders"):
        try:
            out.append({"name": r[0], "centroid": _vec_from_blob(r[1]),
                        "tags": json.loads(r[2] or "[]"),
                        "people": json.loads(r[3] or "[]"),
                        "place": r[4], "date_from": r[5], "date_to": r[6],
                        "pattern": r[7]})
        except Exception:
            continue
    c.close()
    return out

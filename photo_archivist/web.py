"""Read-only Flask web UI: dashboard, browse, search (with thumbnails), detail,
people, folders, review.

No writes ever — scan/undo/enroll stay CLI-only. The server binds 127.0.0.1 by
default and has NO auth: keep it on localhost. Image serving only ever resolves
paths that came from index.db (user input is limited to a hex file_id).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3

from flask import (Flask, abort, render_template, request, send_file,
                   send_from_directory)

_FILE_ID_RE = re.compile(r"^[0-9a-f]{6,64}$")
_PAGE = 24


def _load_cfg(path: str) -> dict:
    if os.path.exists(path):
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _people_maps(cfg: dict) -> tuple[dict, dict]:
    """roles + people maps (config.yaml, overridden by people_library.yaml)."""
    roles: dict = dict(cfg.get("roles") or {})
    people: dict = dict(cfg.get("people") or {})
    if os.path.exists("people_library.yaml"):
        import yaml
        with open("people_library.yaml") as f:
            lib = yaml.safe_load(f) or {}
        roles.update(lib.get("roles") or {})
        people.update(lib.get("people") or {})
    return roles, people


def create_app(config_path: str = "config.yaml") -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    cfg = _load_cfg(config_path)
    org_root = os.path.realpath(cfg.get("organised_dir", "Organised"))
    app.config["ARCH_CFG"] = cfg
    app.config["ARCH_ORG_ROOT"] = org_root
    app.config["ARCH_CONFIG_PATH"] = config_path

    def _db() -> sqlite3.Connection:
        from .core import index as idx
        c = idx.connect(cfg.get("index_db", "index.db"))
        c.row_factory = sqlite3.Row
        return c

    def _resolve_file(rec: sqlite3.Row) -> str | None:
        """Absolute path for a DB record — organised copy (inside org_root)
        preferred, then the absolute source path we recorded ourselves."""
        cand = rec["organised_path"]
        if cand:
            candidates = [os.path.realpath(cand)]
            if not os.path.isabs(cand):
                parts = [p for p in cand.replace("\\", "/").split("/") if p]
                base = os.path.basename(org_root.rstrip(os.sep))
                if parts and parts[0] == base:
                    candidates.append(os.path.join(os.path.dirname(org_root), *parts[1:]))
                    candidates.append(os.path.join(org_root, *parts[1:]))
                elif parts:
                    candidates.append(os.path.join(org_root, *parts))
            for ap in candidates:
                if (ap == org_root or ap.startswith(org_root + os.sep)) \
                        and os.path.isfile(ap):
                    return ap
        src = rec["source_path"]
        if src and os.path.isabs(src):
            ap = os.path.realpath(src)
            if os.path.isfile(ap):
                return ap
        return None

    def _record(file_id: str) -> sqlite3.Row:
        if not _FILE_ID_RE.match(file_id or ""):
            abort(404)
        c = _db()
        try:
            row = c.execute("SELECT * FROM files WHERE file_id=?",
                            (file_id,)).fetchone()
        finally:
            c.close()
        if row is None:
            abort(404)
        return row

    # ---------------- pages ----------------

    @app.route("/")
    def dashboard():
        c = _db()
        try:
            def one(sql, *a):
                return (c.execute(sql, a).fetchone()[0] or 0)
            stats = {
                "files": one("SELECT count(*) FROM files"),
                "images": one("SELECT count(*) FROM files WHERE type='image'"),
                "documents": one("SELECT count(*) FROM files WHERE type='document'"),
                "videos": one("SELECT count(*) FROM files WHERE type='video'"),
                "review": one("SELECT count(*) FROM files WHERE organised_path LIKE '%_Review%'"),
                "folders": one("SELECT count(*) FROM folders"),
                "people_rows": one("SELECT count(*) FROM people"),
                "named": one("SELECT count(DISTINCT name) FROM people "
                             "WHERE name NOT LIKE 'Unknown%'"),
                "unknown": one("SELECT count(*) FROM people WHERE name LIKE 'Unknown%'"),
            }
        finally:
            c.close()
        env = [(exe, shutil.which(exe)) for exe in ("exiftool", "tesseract", "ffprobe")]
        from .core import faces as facemod
        fok, fmsg = facemod.available()
        vc = cfg.get("vision_llm") or {}
        lib_path = (cfg.get("faces") or {}).get("library", "faces_library.json")
        lib_info = "not created yet"
        if os.path.exists(lib_path):
            try:
                with open(lib_path) as f:
                    lib = json.load(f) or {}
                from collections import Counter
                dims = Counter(len(v) for v in lib.values() if isinstance(v, list))
                dim_str = ", ".join(f"{d}-dim x{n}" for d, n in sorted(dims.items()))
                lib_info = f"{os.path.abspath(lib_path)} ({len(lib)} enrolled; {dim_str})"
            except Exception:
                lib_info = f"{os.path.abspath(lib_path)} (unreadable JSON)"
        return render_template("dashboard.html", stats=stats, env=env,
                               fok=fok, fmsg=fmsg, vc=vc, lib_info=lib_info)

    @app.route("/browse")
    def browse():
        page = max(1, request.args.get("page", 1, type=int))
        type_f = request.args.get("type", "")
        sql = "SELECT * FROM files WHERE 1=1"
        args: list = []
        if type_f in ("image", "document", "video"):
            sql += " AND type=?"
            args.append(type_f)
        sql += " ORDER BY taken_at DESC LIMIT ? OFFSET ?"
        args += [_PAGE, (page - 1) * _PAGE]
        c = _db()
        try:
            rows = [dict(r) for r in c.execute(sql, args).fetchall()]
            total = c.execute("SELECT count(*) FROM files").fetchone()[0]
        finally:
            c.close()
        pages = max(1, (total + _PAGE - 1) // _PAGE)
        return render_template("browse.html", rows=rows, page=page, pages=pages,
                               total=total, type_f=type_f)

    @app.route("/search")
    def search_route():
        q = (request.args.get("q") or "").strip()
        hits: list = []
        if q:
            from .core import search as searchmod
            roles, people = _people_maps(cfg)
            hits = searchmod.search(cfg.get("index_db", "index.db"), q, roles,
                                    limit=50, people_map=people)
        return render_template("search.html", q=q, hits=hits)

    @app.route("/file/<file_id>")
    def detail(file_id: str):
        rec = _record(file_id)
        ap = _resolve_file(rec)
        sidecar, sidecar_json = None, None
        if ap and os.path.isfile(ap + ".tags.json"):
            sidecar = ap + ".tags.json"
        elif rec["organised_path"]:
            p = os.path.realpath(rec["organised_path"])
            if os.path.isfile(p + ".tags.json"):
                sidecar = p + ".tags.json"
        if sidecar:
            try:
                with open(sidecar) as f:
                    sidecar_json = json.dumps(json.load(f), indent=2, default=str)
            except Exception:
                sidecar_json = "(unreadable sidecar)"
        c = _db()
        try:
            people = [dict(r) for r in c.execute(
                "SELECT name,source,conf,box FROM people WHERE file_id=?",
                (file_id,)).fetchall()]
        finally:
            c.close()
        return render_template("detail.html", rec=dict(rec), people=people,
                               has_image=ap is not None, sidecar=sidecar_json)

    @app.route("/people")
    def people_page():
        c = _db()
        try:
            named = [dict(r) for r in c.execute(
                "SELECT name, count(*) AS n, round(avg(conf),2) AS ac, "
                "group_concat(DISTINCT source) AS srcs FROM people "
                "WHERE name NOT LIKE 'Unknown%' GROUP BY name ORDER BY n DESC"
            ).fetchall()]
            unknown = c.execute(
                "SELECT count(*) FROM people WHERE name LIKE 'Unknown%'"
            ).fetchone()[0]
        finally:
            c.close()
        enrolled = []
        lib_path = (cfg.get("faces") or {}).get("library", "faces_library.json")
        if os.path.exists(lib_path):
            try:
                with open(lib_path) as f:
                    lib = json.load(f) or {}
                enrolled = sorted(
                    [{"name": k, "dims": len(v) if isinstance(v, list) else "?"}
                     for k, v in lib.items()], key=lambda x: x["name"])
            except Exception:
                pass
        return render_template("people.html", named=named, unknown=unknown,
                               enrolled=enrolled)

    @app.route("/folders")
    def folders_page():
        c = _db()
        try:
            rows = [dict(r) for r in c.execute(
                "SELECT name, tags, people, place, date_from, date_to, pattern "
                "FROM folders ORDER BY name").fetchall()]
        finally:
            c.close()
        for r in rows:
            for k in ("tags", "people"):
                try:
                    r[k] = json.loads(r.get(k) or "[]")
                except Exception:
                    r[k] = []
        return render_template("folders.html", rows=rows)

    @app.route("/review")
    def review_page():
        c = _db()
        try:
            rows = [dict(r) for r in c.execute(
                "SELECT * FROM files WHERE organised_path LIKE '%_Review%' "
                "OR file_id IN (SELECT file_id FROM people WHERE name LIKE 'Unknown%') "
                "ORDER BY taken_at DESC").fetchall()]
            by_file: dict = {}
            for p in c.execute("SELECT file_id, name, conf FROM people "
                               "WHERE name LIKE 'Unknown%'").fetchall():
                by_file.setdefault(p["file_id"], []).append(
                    f"{p['name']} ({p['conf']})")
        finally:
            c.close()
        for r in rows:
            r["unknown_people"] = by_file.get(r["file_id"], [])
        return render_template("review.html", rows=rows)

    # ---------------- images (DB-trusted paths only) ----------------

    @app.route("/thumb/<file_id>")
    def thumb(file_id: str):
        rec = _record(file_id)
        ap = _resolve_file(rec)
        if ap is None:
            return send_from_directory(app.static_folder, "missing.svg",
                                       mimetype="image/svg+xml")
        if not str(rec["mime"] or "").startswith("image/"):
            return send_from_directory(app.static_folder, "file.svg",
                                       mimetype="image/svg+xml")
        try:
            return send_file(ap, mimetype=rec["mime"], conditional=True)
        except Exception:
            return send_from_directory(app.static_folder, "missing.svg",
                                       mimetype="image/svg+xml")

    @app.route("/file/<file_id>/image")
    def full_image(file_id: str):
        rec = _record(file_id)
        ap = _resolve_file(rec)
        if ap is None or not str(rec["mime"] or "").startswith("image/"):
            abort(404)
        return send_file(ap, mimetype=rec["mime"], conditional=True)

    return app




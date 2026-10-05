"""CLI: scan --dry-run | scan --apply --yes | search | undo | review."""
from __future__ import annotations
import json
import os
import yaml
import typer
from .core import pipeline as pipe
from .core import index as idx
from .core import decide as decmod
from .core import organise as orgmod
from .core import report as reportmod
from .core import search as searchmod
from .core.metadata import fs as fsmod

app = typer.Typer(add_completion=False)


def load_cfg(path: str = "config.yaml") -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def iter_source(src: str, exclude: list) -> list[str]:
    out: list[str] = []
    for root, _dirs, files in os.walk(src):
        for fn in files:
            p = os.path.join(root, fn)
            if fsmod.is_excluded(p, exclude):
                continue
            out.append(p)
    return sorted(out)


@app.command()
def scan(source: str, dry_run: bool = True, yes: bool = False,
         config: str = "config.yaml"):
    cfg = load_cfg(config)
    files = iter_source(source, cfg.get("exclude", []))
    typer.echo(f"Found {len(files)} files under {source}")
    records = [pipe.process_file(p, cfg) for p in files]
    # duplicate link via sha
    seen: dict = {}
    for r in records:
        if r["sha256"] in seen:
            seen[r["sha256"]]["duplicates"].append(r["file_id"])
            r["duplicates"].append(seen[r["sha256"]]["file_id"])
        else:
            seen[r["sha256"]] = r
    # folder profiles learned from previous applies (empty -> all new_folder)
    db = cfg.get("index_db", "index.db")
    profiles: list = idx.load_folders(db)
    if profiles:
        typer.echo(f"Loaded {len(profiles)} folder profile(s): "
                   + ", ".join(p["name"] for p in profiles))
    th = cfg.get("thresholds", {}) or {}
    decisions = [decmod.decide(r["vectors"]["text"], set(r["tags"] or []),
                               set(p["name"] for p in r["people"] or []),
                               profiles, th.get("promote", 0.80),
                               th.get("review_low", 0.60)) for r in records]
    rep = reportmod.build_report(records, decisions)
    typer.echo(json.dumps(rep, indent=2, default=str))
    if dry_run:
        typer.echo("DRY RUN — nothing written. Re-run with --no-dry-run --yes to apply.")
        return
    if not yes:
        typer.echo("Refusing to write without --yes (dry run first rule).")
        raise typer.Exit(1)
    # apply: write index + copies + sidecars + undo.log (+ folder profiles)
    organised = cfg.get("organised_dir", "Organised")
    ulog = cfg.get("undo_log", "undo.log")
    c = idx.connect(db)
    copies: list[str] = []
    buckets: dict[str, list] = {}   # folder name -> records filed there
    for r, d in zip(records, decisions):
        if d["action"] == "file" and d.get("folder"):
            folder = d["folder"]            # promoted into a learned folder
        elif d["action"] == "review":
            folder = "_Review"
        else:
            folder = "Inbox"
        dest_dir = os.path.join(organised, folder)
        dest = orgmod.organise_copy(r["source_path"], dest_dir,
                                    hardlink=bool(cfg.get("hardlink", False)))
        copies.append(dest)
        r["organised_path"] = dest
        orgmod.write_sidecar(dest, r)
        idx.upsert_file(c, r)
        buckets.setdefault(folder, []).append(r)
    # learning loop: persist what each folder now looks like for the next scan
    for name, recs in buckets.items():
        idx.upsert_folder(c, idx.profile_from_records(name, recs))
    orgmod.log_undo(ulog, {"copies": copies, "count": len(copies)})
    c.close()
    typer.echo(f"WROTE {len(copies)} copies into {sorted(buckets)}. "
               f"Undo with: archivist undo")


@app.command()
def search(query: str, config: str = "config.yaml", limit: int = 20):
    cfg = load_cfg(config)
    # People library: people_library.yaml wins; fall back to config.yaml keys.
    roles: dict = dict(cfg.get("roles") or {})
    people: dict = dict(cfg.get("people") or {})
    if os.path.exists("people_library.yaml"):
        with open("people_library.yaml") as f:
            lib = yaml.safe_load(f) or {}
            roles.update(lib.get("roles") or {})
            people.update(lib.get("people") or {})
    hits = searchmod.search(cfg.get("index_db", "index.db"), query, roles, limit)
    if not hits:
        typer.echo("No matches. (Try fewer/other words, or run a scan first.)")
    for hit in hits:
        typer.echo(f"{hit['organised_path'] or hit['source_path']} :: {hit['reason']}")


@app.command()
def undo(config: str = "config.yaml"):
    cfg = load_cfg(config)
    removed = orgmod.undo_last(cfg.get("undo_log", "undo.log"))
    typer.echo(f"Removed {len(removed)} files")


@app.command()
def enroll(name: str, photo: str, config: str = "config.yaml"):
    """Register a confirmed face from a photo into faces_library.json (local only)."""
    import json
    cfg = load_cfg(config)
    faces_cfg = cfg.get("faces") or {}
    lib_path = faces_cfg.get("library", "faces_library.json")
    try:
        from .core.vision_local import LocalVision
        vb = LocalVision()
    except Exception as e:
        typer.echo(f"Face detection unavailable: {e}")
        typer.echo("Install local AI first: pip install -e .[ai-local]")
        raise typer.Exit(1)
    vr = vb.describe(photo, {})
    if not vr.face_embeddings:
        typer.echo(f"No face found in {photo}")
        raise typer.Exit(1)
    lib = json.loads(open(lib_path).read()) if os.path.exists(lib_path) else {}
    lib[name] = vr.face_embeddings[0]   # primary (first detected) face
    with open(lib_path, "w") as f:
        json.dump(lib, f)
    typer.echo(f"Enrolled {name} -> {lib_path} "
               f"({len(lib)} face(s) in library, never uploaded)")


if __name__ == "__main__":
    app()

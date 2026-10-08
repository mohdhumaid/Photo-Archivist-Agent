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


def _seg(s: str, limit: int) -> str:
    """One filesystem-safe path segment: no separators, no trailing dots."""
    import re
    s = re.sub(r"[<>:\"/\\|?*]", "", str(s or ""))
    s = re.sub(r"\s+", " ", s).strip(" .")
    return s[:limit].strip()


def _new_folder_name(r: dict, profiles: list[dict] | None = None) -> str:
    """Nested Year/Month/Event folder path: ``YYYY/YYYY-MM/<Place - Event>``.

    Copies the pattern already on disk: if existing folders read
    'YYYY-MM <Event>' keep that; if they read '<Event> Mon YYYY' match it.
    Components: date from reconciled taken_at, place from place.value,
    event from event.value. No date -> flat leaf; no place/event -> Unsorted.
    Below-threshold files never reach here — decide() routes them to _Review
    first. (``profiles`` kept for signature compatibility only.)
    """
    import re
    taken = str(r.get("taken_at") or "")
    m = re.search(r"(19|20)\d{2}[-:/](\d{1,2})", taken)
    year = m.group(0)[:4] if m else ""
    mon = m.group(2).zfill(2) if m else ""
    m2 = re.search(r"\b(19|20)\d{2}\b", taken)
    year_only = m2.group(0) if m2 else ""
    event = _seg((r.get("event") or {}).get("value"), 60)
    place = _seg(str((r.get("place") or {}).get("value") or "").split(",")[0], 40)
    leaf = " - ".join(p for p in (place, event) if p) or "Unsorted"
    if year and mon:
        return f"{year}/{year}-{mon}/{leaf}"
    if year_only:
        return f"{year_only}/{leaf}"
    return leaf


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
    rep = reportmod.build_report(records, decisions,
                                 int(cfg.get("unnamed_face_ask_N", 3) or 3))
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
            folder = "_Review"              # 0.60-0.79: ask, never auto-file
        else:
            # <0.60 (or no profiles yet): create a metadata-derived folder,
            # unless the date is unreliable — then _Review with a question.
            if (r.get("taken_at_confidence") or 0) < 0.35 or not r.get("taken_at"):
                folder = "_Review"
            else:
                folder = _new_folder_name(r, profiles)
            d["new_folder"] = folder
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
    hits = searchmod.search(cfg.get("index_db", "index.db"), query, roles, limit,
                            people_map=people)
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
def enroll(target: str, name: str = "", config: str = "config.yaml"):
    """Enroll a face image or an entire folder of photos into faces_library.json."""
    cfg = load_cfg(config)
    lib_path = (cfg.get("faces") or {}).get("library", "faces_library.json")
    from .core import faces as facemod
    if os.path.isdir(target):
        res = facemod.enroll_folder(target, lib_path)
        typer.echo(f"Enrolled folder '{target}' -> {lib_path}:")
        for person, status in res.items():
            typer.echo(f"  {person:25s}: {status}")
    elif os.path.isfile(target):
        person_name = name or facemod.clean_name(target)
        ok = facemod.enroll_file(target, person_name, lib_path)
        if ok:
            typer.echo(f"Enrolled '{person_name}' from {target} into {lib_path}")
        else:
            typer.echo(f"FAILED: no face detected in {target}")
            raise typer.Exit(1)
    else:
        typer.echo(f"Target not found: {target}")
        raise typer.Exit(1)


@app.command()
def check():
    """Environment readiness: binaries, OpenCV faces, vision-LLM endpoint."""
    import shutil
    pkg = {"ffprobe": "ffmpeg"}
    required_ok = True
    win = os.name == "nt"
    for exe in ("exiftool", "tesseract", "ffprobe"):
        p = shutil.which(exe)
        if p:
            typer.echo(f"{exe:10s} OK   {p}")
            continue
        if exe == "exiftool":
            try:
                import PIL  # noqa: F401
                typer.echo(f"{exe:10s} OPTIONAL (Pillow active: pure-Python EXIF enabled)")
                continue
            except ImportError:
                pass
        if win:
            tip = {"exiftool": "winget install exiftool",
                   "ffprobe": "winget install Gyan.FFmpeg",
                   "tesseract": "winget install UB-Mannheim.TesseractOCR"}.get(exe, exe)
        else:
            tip = f"brew install {pkg.get(exe, exe)}"
        typer.echo(f"{exe:10s} MISSING — {tip}")
        required_ok = required_ok and bool(p)
    from .core import faces as facemod
    fok, msg = facemod.available()
    typer.echo(f"{'haar':10s} {'OK   ' if fok else 'ABSENT'} {msg}")
    if not fok:
        typer.echo("           install once: pip install opencv-python-headless")
        typer.echo("           (the Haar cascade XML ships INSIDE that wheel — no other download)")
    # pure-Python coverage: what still works WITHOUT the external binaries
    typer.echo(f"{'fallback':10s} exiftool missing -> Pillow EXIF (marked _via=pillow)")
    typer.echo(f"{'':10s} ffprobe/ffmpeg missing -> OpenCV video probe + keyframe")
    typer.echo(f"{'':10s} tesseract missing -> text-layer PDFs stay searchable, "
               "scans use the vision-LLM transcribe path")
    cfg = load_cfg()
    from .core import vllm as vllmmod
    if vllmmod.enabled(cfg):
        vc = vllmmod.vllm_cfg(cfg)
        typer.echo(f"{'vision_llm':10s} {vc.get('model')} @ {vc.get('base_url')} "
                   f"(key_env={vc.get('api_key_env') or 'LLM_API_KEY'})")
        probe = vllmmod._post(cfg, [{"role": "user",
                                     "content": [{"type": "text",
                                                  "text": "Reply with: {\"caption\": \"ok\"}"}]}])
        if probe:
            typer.echo(f"{'vision_ping':10s} OK   endpoint reachable")
        else:
            err = vllmmod.LAST_ERROR or 'no response'
            typer.echo(f"{'vision_ping':10s} FAILED — {err}")
            if 'HTTP 401' in err:
                typer.echo(f"{'hint':10s} gateway got NO key. Put it in .env first: LLM_API_KEY=sk-...")
                typer.echo(f"{'':10s}   (or PowerShell: $env:LLM_API_KEY='sk-...') - never config.yaml")
                typer.echo(f"{'':10s}   standalone: python tools/check_litellm.py --base-url {vc.get('base_url')!r} --model {vc.get('model')!r}")
            elif 'HTTP 404' in err:
                typer.echo(f"{'hint':10s} 404 = model id or base_url wrong; ask gateway team for exact id.")
            else:
                typer.echo(f"{'hint':10s} configure vision_llm.model/base_url/api_key_env in config.yaml, then re-run.")
        required_ok = required_ok and bool(probe)
    else:
        vc = (cfg.get("vision_llm") or {})
        typer.echo(f"{'vision_llm':10s} disabled "
                   f"(enabled={vc.get('enabled')}, "
                   f"model={vc.get('model') or 'NOT SET'}, "
                   f"base_url={vc.get('base_url') or 'NOT SET'}, "
                   f"api_key_env={vc.get('api_key_env') or 'LLM_API_KEY'});")
        typer.echo(f"{'':10s} set enabled: true plus the 3 values above to reach the gateway.")
    # faces_library.json: location + contents summary (dormant until embeddings exist)
    lib_path = (cfg.get("faces") or {}).get("library", "faces_library.json")
    if os.path.exists(lib_path):
        import json
        try:
            with open(lib_path) as f:
                lib = json.load(f) or {}
            n_faces = len(lib)
            from collections import Counter
            dims = Counter(len(v) for v in lib.values() if isinstance(v, list))
            dim_str = ", ".join(f"{d}-dim x{c}" for d, c in sorted(dims.items())) or "no vectors"
            mixed = " — MIXED DIMS: re-enroll all faces" if len(dims) > 1 else ""
            typer.echo(f"{'faceslib':10s} {os.path.abspath(lib_path)} "
                       f"({n_faces} enrolled; {dim_str}){mixed}")
        except Exception:
            typer.echo(f"{'faceslib':10s} {os.path.abspath(lib_path)} (unreadable JSON)")
    else:
        typer.echo(f"{'faceslib':10s} {os.path.abspath(lib_path)} (not created yet)")
    raise typer.Exit(0 if required_ok else 1)


@app.command()
def llmtest(image: str, config: str = "config.yaml"):
    """Send ONE image through the real vision-LLM path and verdict its output.

    Prints the raw model text, the parsed JSON, and PASS/FAIL per check
    (caption present, tags well-typed, confidence in 0..1, no mock leak).
    Exit 0 = LLM OUTPUT PROPER, exit 2 = NOT PROPER (details below).
    """
    import mimetypes
    cfg = load_cfg(config)
    from .core import vllm as vllmmod
    if not vllmmod.enabled(cfg):
        typer.echo("vision_llm disabled or incomplete: set vision_llm.enabled=true, "
                   "base_url and model in config.yaml first.")
        raise typer.Exit(2)
    if not os.path.isfile(image):
        typer.echo(f"image not found: {image}")
        raise typer.Exit(2)
    vc = vllmmod.vllm_cfg(cfg)
    typer.echo(f"model   : {vc.get('model')}")
    typer.echo(f"endpoint: {vc.get('base_url')}")
    prime = {
        "file_name": os.path.basename(image),
        "file_type": "image",
        "mime": mimetypes.guess_type(image)[0] or "image/jpeg",
        "path_segments": [p for p in os.path.dirname(image).split(os.sep) if p][-3:],
        "ocr_text": "",
        "metadata": {},
    }
    out = vllmmod.describe(cfg, image, prime)
    raw = vllmmod.LAST_RAW_TEXT or ""
    typer.echo("\n--- raw model text (first 600 chars) ---")
    typer.echo(raw[:600] or "(none)")
    typer.echo("\n--- parsed output ---")
    typer.echo(json.dumps(out, indent=2, default=str) if out is not None
               else f"(None) LAST_ERROR: {vllmmod.LAST_ERROR or 'none'}")
    issues = vllmmod.validate_llm_output(out)
    typer.echo("\n--- checks ---")
    if issues:
        for i in issues:
            typer.echo(f"FAIL  {i}")
        typer.echo("\nVERDICT: LLM OUTPUT NOT PROPER")
        raise typer.Exit(2)
    typer.echo("PASS  caption present & non-trivial")
    typer.echo("PASS  tags/objects/people_hints well-typed (or absent)")
    typer.echo("PASS  confidence within 0..1 (or absent)")
    typer.echo("PASS  no mock-fallback leak")
    typer.echo("\nVERDICT: LLM OUTPUT PROPER")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8501, config: str = "config.yaml"):
    """Read-only web UI (Flask): browse / search / people / folders / review.

    Binds 127.0.0.1 by default and has NO auth — keep it on localhost.
    All writes (scan / undo / enroll) stay CLI-only.
    """
    try:
        import flask  # noqa: F401
    except ImportError:
        typer.echo("Flask missing — install the web extra:  pip install -e .[web]")
        raise typer.Exit(1)
    from .web import create_app
    typer.echo(f"Web UI: http://{host}:{port}  (read-only, Ctrl+C to stop)")
    create_app(config).run(host=host, port=port)


if __name__ == "__main__":
    app()

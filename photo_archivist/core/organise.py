"""Writes: copies (not moves), metadata-preserving, sidecar .tags.json, undo.log."""
from __future__ import annotations
import json
import os
import shutil


def organise_copy(src: str, dest_dir: str, hardlink: bool = False) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, os.path.basename(src))
    base, ext = os.path.splitext(dest)
    i = 1
    while os.path.exists(dest):
        dest = f"{base}_{i}{ext}"
        i += 1
    if hardlink:
        try:
            if os.stat(src).st_dev == os.stat(dest_dir).st_dev:
                os.link(src, dest)
                return dest
        except Exception:
            pass
    shutil.copy2(src, dest)  # preserves metadata byte-for-byte
    return dest


def write_sidecar(dest_path: str, record: dict) -> str:
    side = dest_path + ".tags.json"
    with open(side, "w") as f:
        json.dump(record, f, indent=2, default=str)
    return side


def log_undo(log_path: str, entry: dict) -> None:
    with open(log_path, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def undo_last(log_path: str) -> list[str]:
    """Reverse the last batch: remove copies listed in the most recent line(s)."""
    if not os.path.exists(log_path):
        return []
    with open(log_path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        return []
    last = json.loads(lines[-1])
    removed: list[str] = []
    for p in last.get("copies", []):
        try:
            if os.path.exists(p):
                os.remove(p)
                removed.append(p)
            side = p + ".tags.json"
            if os.path.exists(side):
                os.remove(side)
        except Exception:
            continue
    with open(log_path, "a") as f:
        f.write(json.dumps({"undo_of": last, "removed": removed}) + "\n")
    return removed

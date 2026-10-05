"""Step 3b: filesystem layer + sidecars + filename/path parsing (spec §3.2 tail)."""
from __future__ import annotations
import fnmatch
import json
import os
import re
import subprocess
from pathlib import Path


def fs_stat(path: str) -> dict:
    st = os.stat(path)
    return {
        "size": st.st_size, "mtime": st.st_mtime, "ctime": st.st_ctime,
        "birthtime": getattr(st, "st_birthtime", None),
        "inode": getattr(st, "st_ino", None), "dev": getattr(st, "st_dev", None),
    }


def xattrs(path: str) -> dict:
    """macOS xattr -l; returns {} on Linux/other failures."""
    try:
        out = subprocess.run(["xattr", "-l", path], capture_output=True, text=True, timeout=10)
        return {"raw": out.stdout or "", "where_froms": _parse_where_froms(out.stdout or "")}
    except Exception:
        return {}


def _parse_where_froms(raw: str) -> list[str]:
    m = re.findall(r"kMDItemWhereFroms.*?:\n((?:.*\n)+?)(?:\n\S|\Z)", raw)
    return [s.strip() for s in m] if m else []


def takeout_json(path: str) -> dict | None:
    """Google Takeout / Drive supplemental JSON beside the file."""
    for cand in (path + ".json", path + ".supplemental-metadata.json"):
        if os.path.exists(cand):
            try:
                with open(cand) as f:
                    return json.load(f)
            except Exception:
                return {"_parse_error": cand}
    # <filename>.supplemental-metadata.json pattern with stem match
    p = Path(path)
    for sib in p.parent.glob(p.stem + "*.json"):
        try:
            with open(sib) as f:
                return json.load(f)
        except Exception:
            continue
    return None


def xmp_sidecar(path: str) -> str | None:
    stem = os.path.splitext(path)[0]
    for cand in (stem + ".xmp", stem + ".XMP"):
        if os.path.exists(cand):
            try:
                with open(cand, encoding="utf-8", errors="replace") as f:
                    return f.read()
            except Exception:
                return None
    return None


FILENAME_PATTERNS = [
    ("whatsapp", re.compile(r"IMG-(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})-WA(?P<n>\d+)", re.I)),
    ("pixel", re.compile(r"PXL_(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})_(?P<rest>\d+)", re.I)),
    ("screenshot_mac", re.compile(r"Screenshot (?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2}) at (?P<h>\d{1,2})\.(?P<mi>\d{2})\.(?P<s>\d{2})", re.I)),
    ("nikon", re.compile(r"DSC_(?P<n>\d+)", re.I)),
    ("compact_date", re.compile(r"(?P<y>20\d{2})(?P<m>\d{2})(?P<d>\d{2})_(?P<h>\d{2})(?P<mi>\d{2})(?P<s>\d{2})")),
]


def parse_filename(name: str) -> dict:
    for kind, rx in FILENAME_PATTERNS:
        m = rx.search(name)
        if m:
            d = {"pattern": kind, **m.groupdict()}
            if kind == "whatsapp" and all(k in d for k in ("y", "m", "d")):
                d["embedded_date"] = f"{d['y']}-{d['m']}-{d['d']}"
            return d
    return {}


def path_segments(path: str) -> list[str]:
    parts = [p for p in Path(path).parts if p not in ("/",)]
    return parts[:-1]  # folder segments only


def is_excluded(path: str, exclude: list[str]) -> bool:
    base = os.path.basename(path)
    for pat in exclude or []:
        if fnmatch.fnmatch(base, pat) or fnmatch.fnmatch(path, pat):
            return True
    return False

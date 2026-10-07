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
    ("screenshot_win", re.compile(r"Screenshot[ _-]*\((?P<d>\d{1,2})-(?P<m>\d{1,2})-(?P<y>\d{4})\)", re.I)),
    ("nikon", re.compile(r"DSC_(?P<n>\d+)", re.I)),
    ("compact_date", re.compile(r"(?P<y>20\d{2})(?P<m>\d{2})(?P<d>\d{2})_(?P<h>\d{2})(?P<mi>\d{2})(?P<s>\d{2})")),
]


def parse_filename(name: str) -> dict:
    """Filename-as-metadata: WhatsApp / Pixel / Screenshots / DSC / compact dates.

    Returns {pattern, embedded_date?, embedded_datetime?, ...}. Never guesses
    beyond what the name literally carries.
    """
    for kind, rx in FILENAME_PATTERNS:
        m = rx.search(name)
        if m:
            d = {"pattern": kind, **{k: v for k, v in m.groupdict().items() if v is not None}}
            if kind == "whatsapp" and all(k in d for k in ("y", "m", "d")):
                d["embedded_date"] = f"{d['y']}-{d['m']}-{d['d']}"
            elif kind == "screenshot_mac" and all(k in d for k in ("y", "m", "d")):
                d["embedded_date"] = f"{d['y']}-{d['m']}-{d['d']}"
                try:
                    hh = int(d.get("h", 0) or 0)
                    # mac screenshots use 12h clock without AM/PM in name; keep date only
                    d["embedded_datetime"] = (
                        f"{d['y']}-{d['m']}-{d['d']}T{hh:02d}:{d.get('mi', '00')}:{d.get('s', '00')}")
                except Exception:
                    pass
            elif kind == "screenshot_win" and all(k in d for k in ("y", "m", "d")):
                d["embedded_date"] = f"{d['y']}-{d['m']}-{d['d']}"
            elif kind == "pixel" and all(k in d for k in ("y", "m", "d")):
                d["embedded_date"] = f"{d['y']}-{d['m']}-{d['d']}"
                rest = d.get("rest") or ""
                if len(rest) >= 9:  # HHMMSSmmm
                    d["embedded_datetime"] = (
                        f"{d['y']}-{d['m']}-{d['d']}T{rest[0:2]}:{rest[2:4]}:{rest[4:6]}")
            elif kind == "compact_date" and all(k in d for k in ("y", "m", "d", "h", "mi", "s")):
                d["embedded_date"] = f"{d['y']}-{d['m']}-{d['d']}"
                d["embedded_datetime"] = (
                    f"{d['y']}-{d['m']}-{d['d']}T{d['h']}:{d['mi']}:{d['s']}")
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

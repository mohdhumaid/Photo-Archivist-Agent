"""Step 3a: exiftool extraction. RULE — full dump, all groups/tags, never Pillow-subset."""
from __future__ import annotations
import json
import shutil
import subprocess


def require_exiftool() -> str:
    exe = shutil.which("exiftool")
    if not exe:
        raise RuntimeError(
            "exiftool is required but not installed. Install it (brew install exiftool) — "
            "do not fall back to Pillow-only EXIF."
        )
    return exe


def dump(path: str) -> dict:
    """exiftool -j -G -a -u -n -api largefilesupport=1 -r <path> (single file)."""
    exe = require_exiftool()
    out = subprocess.run(
        [exe, "-j", "-G", "-a", "-u", "-n", "-api", "largefilesupport=1", path],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(out.stdout)
    return data[0] if data else {}


def dump_human(path: str) -> dict:
    """Human-readable composite/date pass: -struct -api QuickTimeUTC=1."""
    exe = require_exiftool()
    out = subprocess.run(
        [exe, "-j", "-G", "-a", "-struct", "-api", "QuickTimeUTC=1", path],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(out.stdout)
    return data[0] if data else {}

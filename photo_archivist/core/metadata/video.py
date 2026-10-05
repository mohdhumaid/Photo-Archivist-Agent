"""Step 3d: video/audio via ffprobe + keyframe extraction."""
from __future__ import annotations
import json
import shutil
import subprocess
import tempfile
import os


def ffprobe(path: str) -> dict:
    exe = shutil.which("ffprobe")
    if not exe:
        return {"_error": "ffprobe not found"}
    out = subprocess.run(
        [exe, "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", path],
        capture_output=True, text=True,
    )
    try:
        return json.loads(out.stdout or "{}")
    except Exception:
        return {"_error": out.stderr[:500]}


def keyframe(path: str, at_seconds: float = 1.0) -> str | None:
    """Extract a single JPEG keyframe to a temp file for vision. Returns temp path."""
    exe = shutil.which("ffmpeg")
    if not exe:
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.close()
    try:
        subprocess.run(
            [exe, "-y", "-v", "error", "-ss", str(at_seconds), "-i", path,
             "-frames:v", "1", "-q:v", "3", tmp.name],
            capture_output=True, timeout=60,
        )
        if os.path.getsize(tmp.name) > 0:
            return tmp.name
        return None
    except Exception:
        return None

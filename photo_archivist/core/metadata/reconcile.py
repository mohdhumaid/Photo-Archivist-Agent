"""Step 3e: reconcile per S3.4 trust order. Never invents facts."""
from __future__ import annotations
from datetime import datetime, timezone
import re


def _get(raw: dict, *names: str):
    for n in names:
        for k, v in raw.items():
            if k.split(":")[-1].lower() == n.lower() or k.lower() == n.lower():
                if v not in (None, "", "0000:00:00 00:00:00"):
                    return v, k
    return None, None


def reconcile_date(raw: dict, filename_info: dict, takeout: dict | None,
                   docmeta: dict, fs: dict, tz_default: str = "Asia/Kolkata") -> dict:
    tz_assumed = f"assumed_{tz_default.lower().replace('/', '_')}"
    v, k = _get(raw, "DateTimeOriginal")
    off, _ = _get(raw, "OffsetTimeOriginal", "OffsetTime")
    if v:
        return {"taken_at": str(v), "source": "exif_datetimeoriginal",
                "confidence": 0.99 if off else 0.9,
                "timezone_source": "exif_offsettime" if off else tz_assumed}
    v, k = _get(raw, "GPSDateStamp", "GPSTimeStamp")
    if v:
        return {"taken_at": str(v), "source": "gps_timestamp", "confidence": 0.9,
                "timezone_source": "utc"}
    if takeout:
        ts = (takeout.get("photoTakenTime") or {})
        if isinstance(ts, dict) and ts.get("timestamp"):
            try:
                dt = datetime.fromtimestamp(int(ts["timestamp"]), tz=timezone.utc)
                return {"taken_at": dt.isoformat(), "source": "takeout_json",
                        "confidence": 0.85, "timezone_source": "utc"}
            except Exception:
                pass
    v, k = _get(raw, "CreateDate")
    if v:
        src = "quicktime_createdate" if "quicktime" in (k or "").lower() else "xmp_createdate"
        return {"taken_at": str(v), "source": src, "confidence": 0.8,
                "timezone_source": "utc_note" if "quick" in src else tz_assumed}
    if filename_info.get("embedded_date"):
        return {"taken_at": filename_info["embedded_date"], "source": "filename_date",
                "confidence": 0.6, "timezone_source": tz_assumed}
    if filename_info.get("pattern") == "compact_date":
        try:
            d = filename_info
            iso = f"{d['y']}-{d['m']}-{d['d']}T{d['h']}:{d['mi']}:{d['s']}"
            return {"taken_at": iso, "source": "filename_datetime", "confidence": 0.7,
                    "timezone_source": tz_assumed}
        except Exception:
            pass
    dm = docmeta or {}
    info = dm.get("info", {}) if isinstance(dm.get("info"), dict) else {}
    if info.get("CreationDate"):
        return {"taken_at": str(info["CreationDate"]), "source": "pdf_creationdate",
                "confidence": 0.7, "timezone_source": "as_stored"}
    core = dm.get("docProps/core.xml", {}) or {}
    if isinstance(core, dict) and core.get("created"):
        return {"taken_at": str(core["created"]), "source": "ooxml_created",
                "confidence": 0.7, "timezone_source": "as_stored"}
    if fs.get("birthtime"):
        dt = datetime.fromtimestamp(fs["birthtime"]).isoformat()
        return {"taken_at": dt, "source": "fs_birthtime", "confidence": 0.3,
                "timezone_source": "local_unreliable"}
    if fs.get("mtime"):
        dt = datetime.fromtimestamp(fs["mtime"]).isoformat()
        return {"taken_at": dt, "source": "fs_mtime", "confidence": 0.2,
                "timezone_source": "local_unreliable", "unreliable": True}
    return {"taken_at": None, "source": None, "confidence": 0.0,
            "timezone_source": "unknown"}

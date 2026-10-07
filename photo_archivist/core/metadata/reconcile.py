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


def _norm_exif_dt(v, subsec=None, off=None) -> str:
    """'2025:07:14 11:32:08' (+ SubSec + Offset) -> ISO."""
    s = str(v).strip().replace(":", "-", 2).replace(" ", "T", 1)
    if subsec not in (None, ""):
        try:
            frac = re.sub(r"\D", "", str(subsec))[:3]
            if frac and "." not in s:
                s += f".{frac}"
        except Exception:
            pass
    if off not in (None, ""):
        m = re.match(r"^([+-])(\d{2}):?(\d{2})?$", str(off).strip())
        if m and not re.search(r"[+-]\d{2}:?\d{2}$", s):
            s += f"{m.group(1)}{m.group(2)}:{m.group(3) or '00'}"
    return s


def reconcile_date(raw: dict, filename_info: dict, takeout: dict | None,
                   docmeta: dict, fs: dict, tz_default: str = "Asia/Kolkata") -> dict:
    tz_assumed = f"assumed_{tz_default.lower().replace('/', '_')}"
    v, k = _get(raw, "DateTimeOriginal")
    sub, _ = _get(raw, "SubSecTimeOriginal", "SubSecTime")
    off, _ = _get(raw, "OffsetTimeOriginal", "OffsetTime", "TimeZoneOffset")
    if v:
        iso = _norm_exif_dt(v, sub, off)
        return {"taken_at": iso, "source": "exif_datetimeoriginal",
                "confidence": 0.99 if off else 0.9,
                "timezone_source": "exif_offsettime" if off else tz_assumed}
    gdate, _ = _get(raw, "GPSDateStamp")
    gtime, _ = _get(raw, "GPSTimeStamp")
    if gdate or gtime:
        try:
            ds, ts = str(gdate or "").strip(), str(gtime or "").strip()
            if ds and re.search(r"\d{4}", ds):
                iso = ds.replace(":", "-", 2)
                if ts:
                    iso += f"T{ts}" if ":" in ts else f"T{ts}"
                if not iso.endswith("Z"):
                    iso += "Z"
                return {"taken_at": iso, "source": "gps_timestamp",
                        "confidence": 0.9, "timezone_source": "utc"}
        except Exception:
            pass
        v2, _ = _get(raw, "GPSDateStamp", "GPSTimeStamp")
        if v2:
            return {"taken_at": str(v2), "source": "gps_timestamp",
                    "confidence": 0.9, "timezone_source": "utc"}
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
        kl = (k or "").lower()
        if "quicktime" in kl:
            return {"taken_at": _norm_exif_dt(v), "source": "quicktime_createdate",
                    "confidence": 0.8, "timezone_source": "utc_note"}
        return {"taken_at": _norm_exif_dt(v), "source": "xmp_createdate",
                "confidence": 0.8, "timezone_source": tz_assumed}
    v, k = _get(raw, "MetadataDate")
    if v:
        return {"taken_at": _norm_exif_dt(v), "source": "xmp_metadatadate",
                "confidence": 0.7, "timezone_source": tz_assumed}
    if filename_info.get("embedded_datetime"):
        return {"taken_at": filename_info["embedded_datetime"],
                "source": "filename_datetime",
                "confidence": 0.7, "timezone_source": tz_assumed}
    if filename_info.get("embedded_date"):
        return {"taken_at": filename_info["embedded_date"], "source": "filename_date",
                "confidence": 0.6, "timezone_source": tz_assumed}
    dm = docmeta or {}
    info = dm.get("info", {}) if isinstance(dm.get("info"), dict) else {}
    if info.get("CreationDate"):
        return {"taken_at": str(info["CreationDate"]), "source": "pdf_creationdate",
                "confidence": 0.7, "timezone_source": "as_stored"}
    if info.get("ModDate") or info.get("ModifyDate"):
        return {"taken_at": str(info.get("ModDate") or info.get("ModifyDate")),
                "source": "pdf_moddate", "confidence": 0.5,
                "timezone_source": "as_stored"}
    core = dm.get("docProps/core.xml", {}) or {}
    if isinstance(core, dict) and core.get("created"):
        return {"taken_at": str(core["created"]), "source": "ooxml_created",
                "confidence": 0.7, "timezone_source": "as_stored"}
    if isinstance(core, dict) and core.get("modified"):
        return {"taken_at": str(core["modified"]), "source": "ooxml_modified",
                "confidence": 0.5, "timezone_source": "as_stored"}
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

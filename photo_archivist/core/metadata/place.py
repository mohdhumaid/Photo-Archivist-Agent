"""Place + sanity (spec S3.4). Appended to reconcile.py companions."""
from __future__ import annotations
import re


def _get2(raw: dict, *names: str):
    for n in names:
        for k, v in raw.items():
            if k.split(":")[-1].lower() == n.lower() or k.lower() == n.lower():
                if v not in (None, "", "0000:00:00 00:00:00"):
                    return v, k
    return None, None


def reconcile_place(raw: dict, path_segs: list, takeout: dict | None) -> dict:
    lat, _ = _get2(raw, "GPSLatitude")
    lon, _ = _get2(raw, "GPSLongitude")
    if lat == 0 and lon == 0:
        lat = lon = None  # 0,0 => null it
    if lat is not None and lon is not None:
        err, _ = _get2(raw, "GPSHPositioningError")
        return {"value": None, "source": "gps_fix", "gps": [float(lat), float(lon)],
                "accuracy_m": err, "confidence": 0.9, "needs_geocode": True}
    for field in ("Country-PrimaryLocationName", "City", "Sub-location", "Province-State"):
        v, k = _get2(raw, field)
        if v:
            return {"value": str(v), "source": f"iptc_{field}", "gps": None,
                    "accuracy_m": None, "confidence": 0.8}
    for field in ("City", "Country", "LocationShown"):
        v, k = _get2(raw, field)
        if v:
            return {"value": str(v), "source": f"xmp_{field}", "gps": None,
                    "accuracy_m": None, "confidence": 0.75}
    if takeout and isinstance(takeout.get("geoData"), dict):
        g = takeout["geoData"]
        if g.get("latitude"):
            return {"value": None, "source": "takeout_geo",
                    "gps": [g["latitude"], g.get("longitude")],
                    "accuracy_m": None, "confidence": 0.8, "needs_geocode": True}
    if path_segs:
        return {"value": " / ".join(path_segs[-2:]), "source": "path",
                "gps": None, "accuracy_m": None, "confidence": 0.4}
    return {"value": None, "source": None, "gps": None,
            "accuracy_m": None, "confidence": 0.0}


def sanity_flags(raw: dict, reconciled_date: dict) -> list:
    flags: list = []
    taken = str(reconciled_date.get("taken_at") or "")
    m = re.search(r"(19|20)\d{2}", taken)
    if m:
        year = int(m.group(0))
        if year < 1990:
            flags.append("suspect_date_pre1990")
        from datetime import datetime as _dt
        if year > _dt.now().year + 1:
            flags.append("suspect_date_future")
    c, _ = _get2(raw, "CreateDate")
    mo, _ = _get2(raw, "ModifyDate")
    if c and mo and str(mo) < str(c):
        flags.append("touched_modify_before_create")
    if not raw:
        flags.append("metadata_stripped")
    return flags

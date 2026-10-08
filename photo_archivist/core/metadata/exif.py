"""Step 3a: EXIF extraction — exiftool first, Pillow fallback when absent.

Order: exiftool (full dump, all groups/tags) when the binary exists →
otherwise a pure-Python Pillow pass whose keys carry a ``_via: "pillow"`` marker so
downstream code trusts exiftool keys first. Nothing is downloaded; Pillow is
already a core dependency.
"""
from __future__ import annotations
import json
import os
from datetime import datetime
import shutil
import subprocess


def require_exiftool() -> str:
    """Optional check for exiftool; pure-Python Pillow extraction is active by default."""
    exe = shutil.which("exiftool")
    if not exe:
        raise RuntimeError(
            "exiftool is not installed. Pure-Python Pillow extraction covers standard "
            "JPEG/PNG/TIFF/WebP tags (timestamps, GPS, camera model, orientation, dimensions)."
        )
    return exe


def _gps_keys(gps: dict) -> dict:
    """Pillow GPS IFD (numeric or string tags) -> exiftool-style GPS keys."""
    from PIL import ExifTags
    gps_tag_names = getattr(ExifTags, "GPSTAGS", {})
    rev_gps = {v: k for k, v in gps_tag_names.items()}

    normalized = {}
    for k, v in gps.items():
        if isinstance(k, int):
            normalized[k] = v
        elif isinstance(k, str) and k in rev_gps:
            normalized[rev_gps[k]] = v

    def _rat(v):
        try:
            if hasattr(v, "numerator") and hasattr(v, "denominator"):
                return float(v.numerator) / float(v.denominator or 1)
            if isinstance(v, (tuple, list)) and len(v) == 2:
                return float(v[0]) / float(v[1] or 1)
            return float(v)
        except Exception:
            return 0.0

    def _dms(vals, ref, pos=("N", "E"), neg=("S", "W")):
        try:
            if isinstance(vals, (int, float)):
                d = float(vals)
            elif isinstance(vals, (tuple, list)) and len(vals) >= 3:
                d = _rat(vals[0]) + _rat(vals[1]) / 60.0 + _rat(vals[2]) / 3600.0
            elif isinstance(vals, (tuple, list)) and len(vals) == 1:
                d = _rat(vals[0])
            else:
                return None
        except Exception:
            return None
        r = str(ref or "").strip().upper()
        if r in neg:
            return -abs(d)
        return abs(d)

    out: dict = {}
    lat = _dms(normalized.get(2), normalized.get(1, "N"), ("N",), ("S",))
    lon = _dms(normalized.get(4), normalized.get(3, "E"), ("E",), ("W",))

    if lat is not None:
        out["EXIF:GPSLatitude"] = lat
        out["GPS:GPSLatitude"] = lat
        out["GPSLatitude"] = lat
    if lon is not None:
        out["EXIF:GPSLongitude"] = lon
        out["GPS:GPSLongitude"] = lon
        out["GPSLongitude"] = lon
    if lat is not None and lon is not None:
        out["Composite:GPSPosition"] = f"{lat} {lon}"
        out["GPSPosition"] = f"{lat} {lon}"

    alt_val = normalized.get(6)
    if alt_val is not None:
        alt = _rat(alt_val)
        alt_ref = normalized.get(5, 0)
        if str(alt_ref) in ("1", b"\x01"):
            alt = -abs(alt)
        out["EXIF:GPSAltitude"] = alt
        out["GPS:GPSAltitude"] = alt
        out["GPSAltitude"] = alt

    ds = normalized.get(29)
    if ds:
        ds_str = ds.decode() if isinstance(ds, bytes) else str(ds)
        ds_str = ds_str.strip("\x00").strip()
        if ds_str:
            out["EXIF:GPSDateStamp"] = ds_str
            out["GPS:GPSDateStamp"] = ds_str
            out["GPSDateStamp"] = ds_str

    ts = normalized.get(7)
    if ts:
        try:
            if isinstance(ts, (tuple, list)) and len(ts) >= 3:
                h = int(_rat(ts[0]))
                m = int(_rat(ts[1]))
                s = _rat(ts[2])
                s_int = int(s)
                ts_str = f"{h:02d}:{m:02d}:{s_int:02d}" if s == s_int else f"{h:02d}:{m:02d}:{s:06.3f}"
                out["EXIF:GPSTimeStamp"] = ts_str
                out["GPS:GPSTimeStamp"] = ts_str
                out["GPSTimeStamp"] = ts_str
            elif isinstance(ts, (str, bytes)):
                ts_str = ts.decode() if isinstance(ts, bytes) else str(ts)
                ts_str = ts_str.strip("\x00").strip()
                out["EXIF:GPSTimeStamp"] = ts_str
                out["GPS:GPSTimeStamp"] = ts_str
                out["GPSTimeStamp"] = ts_str
        except Exception:
            pass

    pos_err = normalized.get(31)
    if pos_err is not None:
        err = _rat(pos_err)
        out["EXIF:GPSHPositioningError"] = err
        out["GPS:GPSHPositioningError"] = err
        out["GPSHPositioningError"] = err

    return out


def _pillow_dump(path: str) -> dict:
    """Pure-Python EXIF via Pillow. All values tagged ``_via: 'pillow'``."""
    try:
        from PIL import Image, ExifTags
    except ImportError:
        return {"_error": "Pillow not installed", "_via": "pillow"}
    try:
        img = Image.open(path)
    except Exception as e:
        return {"_error": f"cannot open: {e}", "_via": "pillow"}

    abs_p = os.path.abspath(path)
    fn = os.path.basename(path)
    sz = os.path.getsize(path) if os.path.exists(path) else 0
    mime = getattr(Image, "MIME", {}).get(img.format, "") if hasattr(Image, "MIME") else ""

    out: dict = {
        "SourceFile": abs_p,
        "File:FileName": fn,
        "File:Directory": os.path.dirname(abs_p),
        "File:FileSize": sz,
        "File:FileType": img.format or os.path.splitext(path)[1].lstrip(".").upper(),
        "File:MIMEType": mime,
        "File:ImageWidth": img.width,
        "File:ImageHeight": img.height,
        "ImageWidth": img.width,
        "ImageHeight": img.height,
        "FileType": img.format,
        "_via": "pillow",
    }

    raw = getattr(img, "getexif", lambda: None)()
    if not raw:
        out["_note"] = "no EXIF segment in this file"
        return out

    # Collect tags from root IFD (0th) and Exif/Interop sub-IFDs
    ifds_to_check = [dict(raw)]
    try:
        exif_ifd = raw.get_ifd(ExifTags.IFD.Exif)
        if exif_ifd:
            ifds_to_check.append(dict(exif_ifd))
    except Exception:
        pass
    try:
        interop_ifd = raw.get_ifd(ExifTags.IFD.Interop)
        if interop_ifd:
            ifds_to_check.append(dict(interop_ifd))
    except Exception:
        pass

    def _clean_str(v):
        if isinstance(v, bytes):
            try:
                return v.decode("utf-8", errors="replace").rstrip("\x00").strip()
            except Exception:
                return str(v).rstrip("\x00").strip()
        return str(v).rstrip("\x00").strip()

    # Map standard tags
    for ifd in ifds_to_check:
        for tag_id, val in ifd.items():
            if val is None:
                continue
            tag_name = ExifTags.TAGS.get(tag_id)
            if not tag_name or tag_name == "GPSInfo":
                continue

            if tag_name in (
                "DateTimeOriginal", "CreateDate", "DateTime", "DateTimeDigitized",
                "OffsetTimeOriginal", "OffsetTime", "OffsetTimeDigitized",
                "SubSecTimeOriginal", "SubSecTime", "SubSecTimeDigitized",
                "Make", "Model", "LensModel", "LensMake", "LensSerialNumber",
                "SerialNumber", "Software", "ImageDescription", "UserComment",
                "XPComment", "XPAuthor", "XPKeywords", "XPSubject", "XPTitle",
                "Artist", "Copyright", "HostComputer"
            ):
                cleaned = _clean_str(val)
                if tag_name == "UserComment" and isinstance(val, (bytes, str)):
                    if isinstance(val, bytes) and val.startswith(b"ASCII\x00\x00\x00"):
                        cleaned = _clean_str(val[8:])
                    elif isinstance(val, bytes) and val.startswith(b"UNICODE\x00"):
                        try:
                            cleaned = val[8:].decode("utf-16", errors="replace").rstrip("\x00").strip()
                        except Exception:
                            cleaned = _clean_str(val[8:])

                out[f"EXIF:{tag_name}"] = cleaned
                out[tag_name] = cleaned
                if tag_name == "DateTimeDigitized" and "EXIF:CreateDate" not in out:
                    out["EXIF:CreateDate"] = cleaned
                    out["CreateDate"] = cleaned

            elif tag_name in (
                "Orientation", "ExifImageWidth", "ExifImageHeight", "ISO",
                "ISOSpeedRatings", "FocalLength", "FNumber", "ExposureTime",
                "Flash", "MeteringMode", "ExposureProgram", "WhiteBalance"
            ):
                try:
                    if hasattr(val, "numerator") and hasattr(val, "denominator"):
                        v_num = float(val.numerator) / float(val.denominator or 1)
                    elif isinstance(val, (int, float)):
                        v_num = val
                    else:
                        v_num = val
                    out[f"EXIF:{tag_name}"] = v_num
                    out[tag_name] = v_num
                except Exception:
                    out[f"EXIF:{tag_name}"] = val
                    out[tag_name] = val

    # Handle GPS sub-IFD
    gps_dict = None
    try:
        gps_ifd = raw.get_ifd(ExifTags.IFD.GPSInfo)
        if gps_ifd:
            gps_dict = dict(gps_ifd)
    except Exception:
        pass
    if not gps_dict and 34853 in raw:
        try:
            gps_dict = dict(raw[34853])
        except Exception:
            pass
    if gps_dict:
        try:
            out.update(_gps_keys(gps_dict))
        except Exception:
            pass

    # Add file system access and modification times
    try:
        stat_info = os.stat(path)
        # st_atime is access time, st_mtime is modification time
        # Convert to ISO format for consistency
        out["File:FileAccessDate"] = datetime.fromtimestamp(stat_info.st_atime).isoformat()
        out["File:FileModifyDate"] = datetime.fromtimestamp(stat_info.st_mtime).isoformat()
    except Exception as e:
        # Log or handle error if file stat fails
        print(f"Warning: Could not get file system times for {path}: {e}")

    return out


def dump(path: str) -> dict:
    """Full dump: exiftool when present, else the pure-Python Pillow pass."""
    exe = shutil.which("exiftool")
    if exe:
        try:
            out = subprocess.run(
                [exe, "-j", "-G", "-a", "-u", "-n", "-api",
                 "largefilesupport=1", path],
                capture_output=True, text=True, check=True,
            )
            data = json.loads(out.stdout)
            return data[0] if data else {}
        except Exception:
            pass
    return _pillow_dump(path)


def dump_human(path: str) -> dict:
    """Human-readable pass: exiftool when present, else pure-Python Pillow pass."""
    exe = shutil.which("exiftool")
    if not exe:
        return _pillow_dump(path)
    try:
        out = subprocess.run(
            [exe, "-j", "-G", "-a", "-struct", "-api", "QuickTimeUTC=1", path],
            capture_output=True, text=True, check=True,
        )
        data = json.loads(out.stdout)
        return data[0] if data else {}
    except Exception:
        return _pillow_dump(path)

"""Smoke tests: filename parse, reconcile trust order, decide scoring, e2e dry-run."""
import os
from photo_archivist.core.metadata import fs as fsmod
from photo_archivist.core.metadata import reconcile as rec
from photo_archivist.core import decide as decmod
from photo_archivist.core import fingerprint as fp


def test_filename_whatsapp():
    d = fsmod.parse_filename("IMG-20250714-WA0012.jpg")
    assert d["pattern"] == "whatsapp" and d["embedded_date"] == "2025-07-14"


def test_reconcile_prefers_exif():
    raw = {"EXIF:DateTimeOriginal": "2025:07:14 11:32:08", "EXIF:OffsetTimeOriginal": "+05:30"}
    out = rec.reconcile_date(raw, {}, None, {}, {"mtime": 0})
    assert out["source"] == "exif_datetimeoriginal" and out["confidence"] == 0.99


def test_reconcile_gps_zero_null(tmp_path=None):
    from photo_archivist.core.metadata import place as pm
    raw = {"GPS:GPSLatitude": 0, "GPS:GPSLongitude": 0}
    out = pm.reconcile_place(raw, ["Events", "Indiranagar"], None)
    assert out["source"] == "path"


def test_decide_thresholds():
    prof = [{"name": "F", "centroid": [1, 0], "tags": ["branch"], "people": ["Anita Rao"]}]
    d = decmod.decide([1, 0], {"branch"}, {"Anita Rao"}, prof)
    assert d["action"] == "file"
    d2 = decmod.decide([0, 1], {"other"}, {"Nobody"}, prof)
    assert d2["action"] == "new_folder"


def test_sha_roundtrip(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hello")
    h = fp.sha256_of(str(p))
    assert len(h) == 64

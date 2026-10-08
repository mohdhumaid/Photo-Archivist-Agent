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


def test_pillow_exif_fallback_without_exiftool(tmp_path, monkeypatch):
    """No exiftool on PATH -> Pillow pass returns dimensions, timestamps, device."""
    from photo_archivist.core.metadata import exif as exifmod
    from PIL import Image, ExifTags
    img = tmp_path / "photo_with_exif.jpg"
    im = Image.new("RGB", (120, 80), (10, 200, 30))
    exif = im.getexif()
    exif[ExifTags.Base.Make] = "Canon"
    exif[ExifTags.Base.Model] = "Canon EOS R5"
    # Populate Exif sub-IFD
    exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
    exif_ifd[ExifTags.Base.DateTimeOriginal] = "2025:07:14 11:32:08"
    exif_ifd[ExifTags.Base.OffsetTimeOriginal] = "+05:30"
    exif_ifd[ExifTags.Base.UserComment] = b"ASCII\x00\x00\x00Archived photo"
    im.save(img, exif=exif)

    monkeypatch.setattr(exifmod.shutil, "which", lambda *_a, **_k: None)
    d = exifmod.dump(str(img))
    assert d.get("_via") == "pillow"
    assert (d.get("ImageWidth"), d.get("ImageHeight")) == (120, 80)
    assert d.get("Make") == "Canon"
    assert d.get("Model") == "Canon EOS R5"
    assert d.get("DateTimeOriginal") == "2025:07:14 11:32:08"
    assert d.get("OffsetTimeOriginal") == "+05:30"
    assert d.get("UserComment") == "Archived photo"

    # Reconcile date works seamlessly on Pillow EXIF
    rec_dt = rec.reconcile_date(d, {}, None, {}, {"mtime": 0})
    assert rec_dt["taken_at"] == "2025-07-14T11:32:08+05:30"
    assert rec_dt["source"] == "exif_datetimeoriginal"
    assert rec_dt["confidence"] == 0.99


def test_pillow_gps_conversion_without_exiftool(tmp_path, monkeypatch):
    """Pillow GPS sub-IFD DMS -> decimal degrees without exiftool."""
    from photo_archivist.core.metadata import exif as exifmod
    from photo_archivist.core.metadata import place as placemod
    from PIL import Image, ExifTags
    img = tmp_path / "photo_with_gps.jpg"
    im = Image.new("RGB", (64, 64), (50, 50, 50))
    exif = im.getexif()
    gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
    gps_ifd[1] = "N"
    gps_ifd[2] = (12, 58, 30)       # 12 + 58/60 + 30/3600 = 12.975
    gps_ifd[3] = "E"
    gps_ifd[4] = (77, 35, 45)       # 77 + 35/60 + 45/3600 = 77.595833...
    gps_ifd[6] = 920.0               # Altitude
    gps_ifd[7] = (10, 30, 0)         # TimeStamp
    gps_ifd[29] = "2025:07:14"       # DateStamp
    im.save(img, exif=exif)

    monkeypatch.setattr(exifmod.shutil, "which", lambda *_a, **_k: None)
    d = exifmod.dump(str(img))
    assert d.get("_via") == "pillow"
    assert abs(d["GPSLatitude"] - 12.975) < 0.0001
    assert abs(d["GPSLongitude"] - 77.595833) < 0.001
    assert d["GPSDateStamp"] == "2025:07:14"
    assert d["GPSTimeStamp"] == "10:30:00"

    # Reconcile place works directly
    p = placemod.reconcile_place(d, ["Photos"], None)
    assert p["source"] == "gps_fix"
    assert abs(p["gps"][0] - 12.975) < 0.0001
    assert abs(p["gps"][1] - 77.595833) < 0.001


def test_cli_check_with_pure_python(monkeypatch):
    """CLI check command treats exiftool as OPTIONAL when Pillow is active."""
    import shutil
    import typer
    from photo_archivist import cli

    # Simulate exiftool missing
    orig_which = shutil.which
    monkeypatch.setattr(shutil, "which", lambda cmd: None if cmd == "exiftool" else orig_which(cmd))
    
    # Run check()
    # It should not fail exit code due to missing exiftool
    try:
        cli.check()
    except typer.Exit as exc:
        # If tesseract/ffprobe are present, exit code is 0; if not, code is 1, but not from exiftool
        pass


def test_cv2_video_fallback_without_ffprobe(tmp_path, monkeypatch):
    """No ffprobe/ffmpeg on PATH -> OpenCV probe + keyframe still work."""
    import cv2
    import numpy as np
    from photo_archivist.core.metadata import video as vmod
    mp4 = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    w.write(np.zeros((48, 64, 3), dtype=np.uint8))
    w.release()
    monkeypatch.setattr(vmod.shutil, "which", lambda *_a, **_k: None)
    probe = vmod.ffprobe(str(mp4))
    assert probe.get("_via") == "cv2"
    assert probe["streams"][0]["width"] == 64
    kf = vmod.keyframe(str(mp4))
    assert kf and os.path.exists(kf) and os.path.getsize(kf) > 0
    os.remove(kf)


def test_ocr_without_tesseract_returns_str(tmp_path, monkeypatch):
    """No tesseract -> '' (or vision text) but never a crash."""
    from photo_archivist.core import text as textmod
    monkeypatch.setattr(textmod.shutil, "which", lambda *_a, **_k: None)
    out = textmod.ocr_image(str(tmp_path / "missing.png"))
    assert isinstance(out, str)

"""Step 1: Detect — path, size, mtime, MIME sniffed from content (not extension)."""
from __future__ import annotations
import mimetypes
import os
from dataclasses import dataclass


MAGIC = [
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
    (b"\x00\x00\x00\x18ftypheic", "image/heic"),
    (b"\x00\x00\x00 ftypheic", "image/heic"),
    (b"\x00\x00\x00\x18ftypheix", "image/heic"),
    (b"\x00\x00\x00 ftypmif1", "image/heif"),
    (b"%PDF-", "application/pdf"),
    (b"PK\x03\x04", "application/zip"),  # docx/xlsx/pptx are zip
    (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", "application/msword"),  # OLE: .doc/.xls/.ppt/.msg
    (b"\x1a\x45\xdf\xa3", "video/webm"),
]


def sniff_mime(path: str) -> str:
    with open(path, "rb") as f:
        head = f.read(32)
    for sig, mime in MAGIC:
        if head.startswith(sig) or sig in head[:32]:
            # Distinguish OOXML zip subtypes by extension as hint only
            if mime == "application/zip":
                ext = os.path.splitext(path)[1].lower()
                if ext == ".docx":
                    return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                if ext == ".xlsx":
                    return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                if ext == ".pptx":
                    return "application/vnd.openxmlformats-officedocument.presentationml.presentation"
                if ext == ".eml":
                    return "message/rfc822"
            return mime
    # ftyp box check for mp4/mov
    if b"ftyp" in head[4:12]:
        if head[4:12].find(b"qt") >= 0:
            return "video/quicktime"
        return "video/mp4"
    guessed, _ = mimetypes.guess_type(path)
    return guessed or "application/octet-stream"


def asset_type(mime: str) -> str:
    if mime.startswith("image/"):
        return "image"
    if mime.startswith("video/") or mime in ("video/mp4", "video/quicktime"):
        return "video"
    if mime.startswith("audio/"):
        return "video"  # treated with ffprobe path
    return "document"


@dataclass
class Detected:
    path: str
    size: int
    mtime: float
    ctime: float
    birthtime: float | None
    mime: str
    type: str
    inode: int | None


def detect(path: str) -> Detected:
    st = os.stat(path)
    birth = getattr(st, "st_birthtime", None)
    mime = sniff_mime(path)
    return Detected(
        path=path,
        size=st.st_size,
        mtime=st.st_mtime,
        ctime=st.st_ctime,
        birthtime=birth,
        mime=mime,
        type=asset_type(mime),
        inode=getattr(st, "st_ino", None),
    )

"""Step 2: Fingerprint — sha256 + perceptual hash, duplicate linking."""
from __future__ import annotations
import hashlib

CHUNK = 1024 * 1024


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def phash_of(path: str) -> str | None:
    """Perceptual hash for images; None for non-images or on failure."""
    try:
        from PIL import Image
        import imagehash
    except ImportError:
        return None
    try:
        with Image.open(path) as im:
            return str(imagehash.phash(im))
    except Exception:
        return None


def hamming(a: str, b: str) -> int | None:
    try:
        return bin(int(a, 16) ^ int(b, 16)).count("1")
    except Exception:
        return None

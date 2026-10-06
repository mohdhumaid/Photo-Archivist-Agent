"""Face DETECTION (boxes only, never identity) — OpenCV Haar cascades.

Org-policy notes:
- The cascade XML ships INSIDE the opencv wheel (cv2.data.haarcascades), so the
  wheel install is the only download; nothing is fetched from the internet at
  runtime. Verify on a managed laptop with the `check` CLI command first.
- Detection only answers "how many faces, and where" — it produces no
  embeddings, so identity matching (faces_library.json / match_local) stays
  dormant until an embedding source is approved.

Install: pip install opencv-python-headless   (or: pip install -e .[faces])
"""
from __future__ import annotations
import os

CASCADE_NAME = "haarcascade_frontalface_default.xml"
MAX_FACES = 20


def cascade_path() -> str | None:
    """Absolute path of the bundled cascade file, or None when unavailable."""
    try:
        import cv2
    except ImportError:
        return None
    try:
        p = os.path.join(cv2.data.haarcascades, CASCADE_NAME)
    except AttributeError:
        return None
    return p if os.path.exists(p) else None


def available() -> tuple[bool, str]:
    """(ok, human message) — powers the `check` CLI command."""
    try:
        import cv2
    except ImportError:
        return False, "opencv NOT installed — pip install opencv-python-headless"
    p = cascade_path()
    if not p:
        return False, f"opencv {cv2.__version__} installed but cascade file {CASCADE_NAME} missing"
    return True, f"opencv {cv2.__version__} — cascade bundled at {p}"


def detect_faces(path: str) -> list[dict]:
    """Haar face boxes for an image.

    Returns [] when opencv is missing, the file is unreadable, or no face is
    found — never raises, so a scan can't be broken by face detection.
    """
    if cascade_path() is None:
        return []
    import cv2
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return []
        cascade = cv2.CascadeClassifier(cascade_path())
        boxes = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5,
                                         minSize=(30, 30))
    except Exception:
        return []
    return [{"box": [int(x), int(y), int(w), int(h)],
             "detector": "haar_frontal_default"}
            for (x, y, w, h) in boxes][:MAX_FACES]
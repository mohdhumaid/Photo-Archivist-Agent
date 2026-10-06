"""Face DETECTION and local template ENROLLMENT / MATCHING — OpenCV Haar cascades.

Org-policy notes:
- The cascade XML ships INSIDE the opencv wheel (cv2.data.haarcascades), so the
  wheel install is the only download; nothing is fetched from the internet at
  runtime. Verify on a managed laptop with the `check` CLI command first.
- Face template embeddings (64x64 equalized grayscale crop vector, L2-normalized)
  are computed locally via OpenCV/numpy with zero external neural network model
  downloads.
- All biometrics are stored locally in faces_library.json and never uploaded.

Install: pip install opencv-python-headless   (or: pip install -e .[faces])
"""
from __future__ import annotations
import json
import os
import re

CASCADE_NAME = "haarcascade_frontalface_default.xml"
MAX_FACES = 20

EMBED_SIZE = 64          # 64x64 = 4096-dim template vector
EMBED_METHOD = "haar_crop64_eq_v1"   # recorded in docs; not a downloaded model

# Known AU Bank & executive normalization
KNOWN_NAMES = {
    "uttamtibrewal": "Uttam Tibrewal",
    "akhilverma": "Akhil Verma",
    "sanjayagarwal": "Sanjay Agarwal",
    "anmolpadhye": "Anmol Padhye",
    "yogeshjain": "Yogesh Jain",
    "yogeshsoni": "Yogesh Soni",
}


def clean_name(filename: str) -> str:
    """Normalize a photo filename into a clean Person Name."""
    base = os.path.splitext(os.path.basename(filename))[0].strip()
    norm = re.sub(r"[-_ ]+", "", base).lower()
    if norm in KNOWN_NAMES:
        return KNOWN_NAMES[norm]
    # Split camelCase
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", base)
    s = re.sub(r"[-_]+", " ", s)
    return s.title().strip()


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


EMBED_SIZE = 64          # 64x64 = 4096-dim template vector
EMBED_METHOD = "haar_crop64_eq_v1"   # recorded in docs; not a downloaded model


def crop_vector(path: str, box: list, size: int = EMBED_SIZE) -> list[float] | None:
    """Policy-compliant template embedding for ONE face box.

    Grayscale crop -> resize to size x size -> histogram equalization ->
    flatten -> L2 normalize. Pure numpy math on already-installed OpenCV:
    no pretrained weights are loaded, so nothing is downloaded and nothing
    leaves the machine. Accuracy is template-level (frontal, consistent
    photos work best) — tune faces.match_threshold to your library.
    """
    if cascade_path() is None:
        return None
    import cv2
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        x, y, w, h = (int(v) for v in box[:4])
        crop = img[y:y + h, x:x + w]
        if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            return None
        crop = cv2.resize(crop, (size, size))
        crop = cv2.equalizeHist(crop)
        vec = crop.astype("float64").flatten()
        norm = float((vec * vec).sum()) ** 0.5 or 1.0
        return [float(v) / norm for v in vec]
    except Exception:
        return None


def embed_largest_face(path: str) -> list[float] | None:
    """Detect faces, embed the largest one. None when no face/opencv."""
    boxes = detect_faces(path)
    if not boxes:
        return None
    biggest = max(boxes, key=lambda b: b["box"][2] * b["box"][3])
    return crop_vector(path, biggest["box"])


def enroll_file(path: str, name: str, lib_path: str = "faces_library.json") -> bool:
    """Enroll a single face image into faces_library.json."""
    vec = embed_largest_face(path)
    if not vec:
        return False
    lib: dict = {}
    if os.path.exists(lib_path):
        try:
            with open(lib_path) as f:
                lib = json.load(f) or {}
        except Exception:
            lib = {}
    lib[name] = vec
    with open(lib_path, "w") as f:
        json.dump(lib, f, indent=2)
    return True


def enroll_folder(folder_path: str, lib_path: str = "faces_library.json") -> dict[str, str]:
    """Enroll all photos in a folder into faces_library.json, using cleaned filename as name.

    Returns dict mapping {person_name: status_string}.
    """
    results: dict[str, str] = {}
    if not os.path.exists(folder_path):
        return results

    lib: dict = {}
    if os.path.exists(lib_path):
        try:
            with open(lib_path) as f:
                lib = json.load(f) or {}
        except Exception:
            lib = {}

    for fn in sorted(os.listdir(folder_path)):
        if fn.startswith(".") or not fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")):
            continue
        p = os.path.join(folder_path, fn)
        name = clean_name(fn)
        vec = embed_largest_face(p)
        if vec:
            lib[name] = vec
            results[name] = f"OK ({len(vec)} dims from {fn})"
        else:
            results[name] = f"FAILED: no face detected in {fn}"

    with open(lib_path, "w") as f:
        json.dump(lib, f, indent=2)

    return results
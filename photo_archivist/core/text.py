"""Step 4: Text — text layer first; OCR only when no text layer exists."""
from __future__ import annotations
import os


def pdf_text(path: str) -> tuple[str, bool]:
    """Returns (text, has_layer)."""
    try:
        from pypdf import PdfReader
        r = PdfReader(path)
        texts = [(p.extract_text() or "") for p in r.pages]
        joined = "\n".join(texts).strip()
        return joined[:20000], bool(joined)
    except Exception:
        return "", False


def ooxml_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".docx":
            from docx import Document
            d = Document(path)
            return "\n".join(p.text for p in d.paragraphs)[:20000]
        if ext == ".xlsx":
            import openpyxl
            wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
            out: list[str] = []
            for ws in wb.worksheets:
                out.append(f"# {ws.title}")
                for row in ws.iter_rows(values_only=True):
                    out.append(" | ".join("" if v is None else str(v) for v in row))
            return "\n".join(out)[:20000]
    except Exception:
        return ""
    return ""


def ocr_image(path: str) -> str:
    """OCR for scanned docs / signage. Called only when no text layer exists.

    Uses the `tesseract` binary directly via subprocess to avoid heavy
    pandas/numpy dependency chains (pytesseract pulls pandas -> pyarrow
    which breaks on NumPy 2.x in some envs).
    """
    import shutil
    import subprocess
    exe = shutil.which("tesseract")
    if not exe:
        return ""
    try:
        out = subprocess.run([exe, path, "stdout", "-l", "eng"],
                             capture_output=True, text=True, timeout=120)
        return (out.stdout or "")[:20000]
    except Exception:
        return ""


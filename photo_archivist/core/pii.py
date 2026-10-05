"""PII flags — flag, don't redact, don't delete."""
from __future__ import annotations
import re

PATTERNS = {
    "pan_like": re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),
    "aadhaar_like": re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
    "phone_in": re.compile(r"(\+91[\s-]?)?[6-9]\d{9}"),
    "account_like": re.compile(r"\b\d{9,18}\b"),
}


def scan_text(*texts: str) -> list[str]:
    flags: list[str] = []
    blob = "\n".join(t or "" for t in texts if t)
    if not blob:
        return flags
    for name, rx in PATTERNS.items():
        if rx.search(blob):
            flags.append(f"{name}_in_text")
    return flags

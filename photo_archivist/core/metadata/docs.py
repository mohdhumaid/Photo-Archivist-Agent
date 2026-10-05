"""Step 3c: document metadata — PDF / OOXML / OLE / email / zip (spec §3.3)."""
from __future__ import annotations
import os
import zipfile
import xml.etree.ElementTree as ET


def _xml_text(xml_bytes: bytes, tag_suffix: str) -> dict:
    try:
        root = ET.fromstring(xml_bytes)
    except Exception:
        return {}
    out: dict = {}
    for el in root.iter():
        if el.tag.endswith(tag_suffix) and el.text and el.text.strip():
            out[tag_suffix] = el.text.strip()
    return out


def ooxml_props(path: str) -> dict:
    """Read docProps/core.xml, app.xml, custom.xml directly from the zip."""
    out: dict = {}
    try:
        with zipfile.ZipFile(path) as z:
            for name in ("docProps/core.xml", "docProps/app.xml", "docProps/custom.xml"):
                try:
                    data = z.read(name)
                except KeyError:
                    continue
                try:
                    root = ET.fromstring(data)
                except Exception:
                    out[name] = {"_parse_error": True}
                    continue
                vals: dict = {}
                for el in root.iter():
                    tag = el.tag.split("}")[-1]
                    if el.text and el.text.strip():
                        vals[tag] = el.text.strip()
                # custom.xml: value lives in nested vt:* element
                if not vals:
                    for el in root.iter():
                        for child in el:
                            if child.text and child.text.strip():
                                vals[el.tag.split("}")[-1]] = child.text.strip()
                out[name] = vals
            # headers/footers carry ref numbers; sheet names for xlsx
            try:
                out["_members"] = z.namelist()[:200]
            except Exception:
                pass
    except Exception as e:
        out["_error"] = str(e)
    return out


def pdf_meta(path: str) -> dict:
    out: dict = {}
    try:
        from pypdf import PdfReader
        r = PdfReader(path)
        try:
            info = dict(r.metadata or {})
            out["info"] = {str(k): str(v) for k, v in info.items()}
        except Exception:
            pass
        try:
            out["pages"] = len(r.pages)
            # text layer present?
            sample = "".join([(r.pages[i].extract_text() or "") for i in range(min(3, len(r.pages)))])
            out["has_text_layer"] = bool(sample.strip())
        except Exception:
            pass
        try:
            out["attachments"] = list((r.attachments or {}).keys()) if hasattr(r, "attachments") else []
        except Exception:
            pass
    except Exception as e:
        out["_error"] = str(e)
    # XMP / DocumentID via pikepdf when available
    try:
        import pikepdf
        with pikepdf.open(path) as pdf:
            with pdf.open_metadata() as meta:
                out["xmp"] = {k: str(v) for k, v in meta.items()}
            try:
                trailer = pdf.trailer
                out["trailer_id"] = str(trailer.get("/ID", ""))
            except Exception:
                pass
    except Exception:
        pass
    return out


def ole_meta(path: str) -> dict:
    """Legacy .doc/.xls/.ppt summary streams + .msg basics."""
    out: dict = {}
    try:
        import olefile
        if olefile.isOleFile(path):
            ole = olefile.OleFileIO(path)
            try:
                meta = ole.get_metadata()
                out = {k: str(v) for k, v in vars(meta).items() if not k.startswith("_")}
            finally:
                ole.close()
    except Exception as e:
        out["_error"] = str(e)
    return out


def email_meta(path: str) -> dict:
    out: dict = {}
    try:
        import email
        from email import policy
        with open(path, "rb") as f:
            msg = email.message_from_binary_file(f, policy=policy.default)
        for h in ("From", "To", "Cc", "Date", "Subject", "Message-ID", "In-Reply-To", "References"):
            v = msg.get(h)
            if v:
                out[h] = str(v)
        out["attachments"] = [p.get_filename() or "" for p in msg.iter_attachments()]
    except Exception as e:
        out["_error"] = str(e)
    return out


def document_metadata(path: str, mime: str) -> dict:
    ext = os.path.splitext(path)[1].lower()
    if mime == "application/pdf" or ext == ".pdf":
        return {"kind": "pdf", **pdf_meta(path)}
    if "openxmlformats" in mime or ext in (".docx", ".xlsx", ".pptx"):
        return {"kind": "ooxml", **ooxml_props(path)}
    if ext in (".doc", ".xls", ".ppt", ".msg"):
        return {"kind": "ole", **ole_meta(path)}
    if ext in (".eml", ".msg") or mime == "message/rfc822":
        return {"kind": "email", **email_meta(path)}
    if ext == ".zip":
        try:
            with zipfile.ZipFile(path) as z:
                infos = z.infolist()
                return {"kind": "zip", "members": [i.filename for i in infos[:500]],
                        "member_count": len(infos)}
        except Exception as e:
            return {"kind": "zip", "_error": str(e)}
    return {}

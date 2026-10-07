"""Generic OpenAI-compatible vision chat client (local vLLM or LiteLLM gateway).

One protocol serves both endpoints — they differ only in config:

  self-hosted vLLM (Qwen3-VL, video+image, usually no key):
    vision_llm.base_url: "http://<host>:8000/v1/chat/completions"
    vision_llm.model: "Qwen/Qwen3-VL-8B-Thinking-FP8"

  LiteLLM gateway (Claude etc, Bearer key):
    vision_llm.base_url: "https://<gateway>/apillmgov/v1/chat/completions"
    vision_llm.model: "claude-sonnet-4-6"
    vision_llm.api_key_env: LLM_API_KEY

Images go as base64 image_url parts; video as file:// video_url part
(vLLM style) or as an extracted keyframe image (portable default).
Any failure returns None — pipeline falls back to mock, never crashes.
"""
from __future__ import annotations
import base64
import mimetypes
import os
import re

# Single-word guesses a model emits with no real evidence. NEVER accepted
# as a location value (the Office/Home problem) — model is also instructed
# never to emit them; this is the code-side backstop.
GENERIC_LOCATION_WORDS = {
    "office", "home", "indoor", "outdoor", "inside", "outside",
    "building", "room", "hall", "branch", "bank", "store", "shop",
}

SYSTEM_PROMPT = """You are the Photo Archivist vision expert inside a bank's
photo/document archiving pipeline. No conversation — one JSON answer.

You receive ONE file (photo, video frame, or scanned document) plus text
context: file name, folder path, OCR text, metadata date/place/tags.
Describe ONLY what you can actually see.

RULES
1. LOCATION: metadata already carries the best place guess (metadata_json).
   Confirm/refine it ONLY from visible evidence: signage, name boards,
   banners, landmarks, uniforms, letterheads. Evidence must name the place.
   NEVER output a generic location (Office, Home, indoor, building, branch,
   bank). Cannot name a specific place -> return null with confidence <= 0.4
   and name the missing evidence.
2. PHOTO DESCRIPTION: 1-2 sentences on setting/background, each visible
   person's activity and mood (smiling cutting ribbon, seated presenting),
   plus notable objects. No identities — never guess a name from a face.
3. VISIBLE TEXT: transcribe banners/boards/cheques/letterheads verbatim.
4. Thin evidence -> low confidence (<=0.4). English only. Tags lowercase,
   max 10, 3 words max each.
5. Flag PII only when clearly visible/printed (account, PAN, Aadhaar, phone).

OUTPUT — one JSON object only, no markdown, no prose outside it:
{"caption": "...", "scene": "...", "event_type": "...", "objects": [],
 "visible_text": "...", "tags": [], "people_hints": [],
 "people_activity": "...", "mood": "...", "background": "...",
 "location_guess": {"value": null, "evidence": "...", "confidence": 0.0},
 "pii_flags": [], "confidence": 0.0, "evidence": "..."}"""


def vllm_cfg(cfg: dict | None) -> dict:
    return (cfg or {}).get("vision_llm") or {}


def enabled(cfg: dict | None) -> bool:
    c = vllm_cfg(cfg)
    return bool(c.get("enabled")) and bool(c.get("base_url")) and bool(c.get("model"))


def _load_dotenv(path: str = ".env") -> None:
    """Read .env (key=val) into os.environ. .env sits next to config.yaml."""
    candidates = [path]
    here = os.path.abspath(os.getcwd())
    for _ in range(3):                       # walk up a few dirs for safety
        here = os.path.dirname(here)
        candidates.append(os.path.join(here, ".env"))
    for cand in candidates:
        if os.path.isfile(cand):
            _read_env(cand)
            return


def _read_env(path: str) -> None:
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k = k.strip()
                v = v.strip()
                if len(v) >= 2 and ((v[0] == """ and v[-1] == """) or
                                     (v[0] == "'" and v[-1] == "'")):
                    v = v[1:-1]
                os.environ.setdefault(k, v)
    except OSError:
        pass


def _secret(c: dict) -> str:
    # 1) .env file next to config.yaml  2) real env var  3) config fallback
    _load_dotenv()
    env_var = str(c.get("api_key_env") or "").strip()
    return os.environ.get(env_var, "") or str(c.get("api_key") or "")


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


def _extract_json(text: str) -> dict | None:
    """Parse vision-LLM output: tolerate fences and prose around the JSON."""
    if not text or not isinstance(text, str):
        return None
    t = re.sub(r"```(?:json)?", "", text).strip()
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end <= start:
        return None
    import json as _json
    for cand in (t[start:end + 1], t):
        try:
            out = _json.loads(cand)
            if isinstance(out, dict):
                return out
        except Exception:
            continue
    return None


def build_messages(cfg: dict | None, path: str, prime: dict) -> tuple:
    """OpenAI-style messages. Returns (messages, keyframe_to_cleanup|None)."""
    c = vllm_cfg(cfg)
    meta = prime.get("metadata") or {}
    ctx = [
        f"file_name: {prime.get('file_name')}",
        f"file_type: {prime.get('file_type')} ({prime.get('mime')})",
        f"path: {' / '.join(prime.get('path_segments') or [])}",
        f"metadata date: {meta.get('taken_at')} (from {meta.get('taken_at_source')})",
        f"metadata place guess: {meta.get('place')}",
        f"metadata tags: {', '.join((meta.get('tags') or [])[:20])}",
        f"ocr_text: {(prime.get('ocr_text') or '')[:1500]}",
    ]
    content: list = [{"type": "text",
                      "text": "Analyze this asset, return the JSON.\n" + "\n".join(ctx)}]
    tmp_frame: str | None = None
    ftype = prime.get("file_type") or ""
    cap_mb = float(c.get("max_image_mb", 10))
    try:
        too_big = os.path.getsize(path) > cap_mb * 1024 * 1024
    except OSError:
        too_big = False
    if ftype == "video" and c.get("video_mode", "keyframe") == "file_url":
        content.append({"type": "video_url", "video_url": {"url": f"file://{path}"}})
    elif ftype == "video":
        try:
            from .metadata import video as videomod
            tmp_frame = videomod.keyframe(path)
        except Exception:
            tmp_frame = None
        if tmp_frame:
            durl = _data_url(tmp_frame)
            if durl:
                content.append({"type": "text",
                                "text": "Attached frame is a 1s keyframe of the video."})
                content.append({"type": "image_url", "image_url": {"url": durl}})
    elif str(prime.get("mime") or "").startswith("image/") or ftype == "image":
        if too_big:
            content.append({"type": "image_url", "image_url": {"url": f"file://{path}"}})
        else:
            try:  # corrupt/unreadable image -> text-first, never crash
                from PIL import Image as _PILImage
                with _PILImage.open(path) as _im:
                    _im.verify()
                durl = _data_url(path, str(prime.get("mime") or ""))
            except Exception:
                durl = None
            if durl:
                content.append(_image_part(durl))
    # documents: text-first, no image bytes
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content}]
    return messages, tmp_frame


LAST_ERROR: str = ""


def _post(cfg: dict | None, messages: list) -> dict | None:
    """POST the chat payload. Failures stash a reason in LAST_ERROR."""
    global LAST_ERROR
    LAST_ERROR = ""
    c = vllm_cfg(cfg)
    url = str(c.get("base_url", "")).rstrip("/")
    payload = {"model": c.get("model"),
               "temperature": float(c.get("temperature", 0.2)),
               "max_tokens": int(c.get("max_tokens", 1024)),
               "messages": messages}
    headers = {"Content-Type": "application/json"}
    key = _secret(c)
    # 401s are the gateway to a missing/empty Bearer key; the
    # caller (cli.check / check_litellm.py) owns the remediation.
    if key is None or key == "":
        _hint = c.get('api_key_env') or 'LLM_API_KEY'
        LAST_ERROR = (f"no Bearer key sent (api_key_env={_hint} unset or empty); "
                      f"set it in your .env (LLM_API_KEY=<token>) or shell, e.g. "
                      f"PowerShell: $env:LLM_API_KEY='sk-...'")
    else:
        headers["Authorization"] = f"Bearer {key}"
    try:
        import requests
        r = requests.post(url, json=payload, headers=headers,
                          timeout=int(c.get("timeout", 120)))
        if r.status_code >= 400:
            body = (r.text or "")[:300]
            LAST_ERROR = f"HTTP {r.status_code}: {body}"
            return None
        try:
            return r.json()
        except Exception as e:
            LAST_ERROR = f"non-JSON 200 response: {e}"
            return None
    except Exception as e:
        if not LAST_ERROR:
            LAST_ERROR = f"{type(e).__name__}: {e}"
        return None


def _message_text(resp: dict | None) -> str | None:
    try:
        ch = (resp or {}).get("choices") or []
        msg = (ch[0] or {}).get("message") or {}
        content = msg.get("content")
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content
                               if isinstance(p, dict))
        return _strip_think(str(content or ""))
    except Exception:
        return None


def scrub_location_guess(out: dict | None) -> dict | None:
    """Backstop: generic single-word guesses become null, never a place."""
    if not isinstance(out, dict):
        return None
    loc = out.get("location_guess") or {}
    val = str(loc.get("value") or "").strip()
    if not val or val.lower() == "null":
        return {"value": None, "evidence": loc.get("evidence") or "",
                "confidence": 0.0}
    if val.lower().strip(" .,") in GENERIC_LOCATION_WORDS:
        return {"value": None, "evidence": f"generic label rejected: {val}",
                "confidence": 0.0, "rejected": val}
    try:
        conf = min(max(float(loc.get("confidence", 0.5)), 0.0), 1.0)
    except (TypeError, ValueError):
        conf = 0.5
    return {"value": val, "evidence": str(loc.get("evidence") or ""),
            "confidence": conf}


def describe(cfg: dict | None, path: str, prime: dict) -> dict | None:
    """Build messages -> POST -> parse JSON (None on any failure)."""
    if not enabled(cfg):
        return None
    messages, tmp_frame = build_messages(cfg, path, prime)
    try:
        resp = _post(cfg, messages)
        if resp is None and "image_url" in (LAST_ERROR or ""):
            flat = []  # some gateways want {"type":"image_url","url":...}
            for m in messages:
                if not isinstance(m.get("content"), list):
                    flat.append(m)
                    continue
                parts = []
                for pt in m["content"]:
                    if pt.get("type") == "image_url":
                        url = (pt.get("image_url") or {}).get("url", "")
                        parts.append({"type": "image_url", "url": url})
                    else:
                        parts.append(pt)
                flat.append({"role": m.get("role"), "content": parts})
            resp = _post(cfg, flat)
        out = _extract_json(_message_text(resp) or "")
        if not isinstance(out, dict) or not out.get("caption"):
            return None
        out["location_guess"] = scrub_location_guess(out)
        return out
    finally:
        if tmp_frame:
            try:
                os.remove(tmp_frame)
            except OSError:
                pass

    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


def _data_url(path: str, mime: str = "") -> str | None:
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        mt = mime or mimetypes.guess_type(path)[0] or "image/jpeg"
        return f"data:{mt};base64,{b64}"
    except OSError:
        return None


def _image_part(durl: str) -> dict:
    """OpenAI-style image part. Qwen3-VL gateways that reject the nested
    ``{"image_url": {"url": ...}}`` envelope accept the flat ``{"url": ...}``
    form - describe() retries with this shape on an explicit envelope error."""
    return {"type": "image_url", "image_url": {"url": durl}}

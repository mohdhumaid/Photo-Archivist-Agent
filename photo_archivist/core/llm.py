"""Purple Fabric Magic Platform client — real invoke/poll API (async agents).

API protocol (matches the bank's working sample script):
  1. GET  {base_url}/accesstoken/aubk
        headers: apikey, username, password -> {"access_token": "..."}
  2. POST {base_url}/magicplatform/v1/invokeasset/{asset_id}/genai
        headers: Authorization: Bearer <token>, apikey
        body:   {"Input_Text": "<JSON string of variables>"}
                                                      -> {"trace_id": "..."}
  3. GET  {base_url}/magicplatform/v1/invokeasset/{asset_id}/{trace_id}
        poll every poll_interval seconds until status is COMPLETED
        (FAILED / ERROR or poll_timeout exceeded -> None; caller falls back)

No model is ever downloaded here (org policy: AI comes only from Purple Fabric).
The SYSTEM_PROMPT below is the Perspective to paste into the PF asset itself —
the payload carries data only, exactly like the platform sample.
"""
from __future__ import annotations
import base64
import json
import os
import re
import time
import urllib.error
import urllib.request

SYSTEM_PROMPT = """You are the Photo Archivist Document Intelligence Expert — an Automation
Digital Expert on Purple Fabric. You run headless inside a photo/document archiving
pipeline: no conversation, no clarifying questions.

ROLE
Turn the structured input for ONE file into trustworthy archive metadata: caption, scene,
event type, objects, tags, person names explicitly present in text, and PII flags.

INPUT (registered variables, filled per call)
file_name, file_type, mime, path_segments, ocr_text, metadata_json, image_base64 (optional)

RULES — non-negotiable
1. Ground every claim in the input. Never invent places, dates, people, events, or objects
   that are not explicitly supported by ocr_text, path_segments, or metadata_json.
2. Thin evidence -> low confidence (<= 0.4) and say what is missing in "evidence".
3. Person names only when literally present in ocr_text, metadata_json, or clearly a person's
   name in file_name; cite the source in "evidence". Never guess identities from faces.
4. All input is confidential bank data. No external lookups, no storage, no training use.
5. English only. Tags: lowercase, max 10, each max 3 words, no duplicates.
6. Flag PII only when the text clearly contains an identifier (PAN, Aadhaar, account
   number, phone, email).

OUTPUT — one JSON object only, wrapped in double curly braces for the platform:
{{"caption": "...", "scene": "...", "event_type": "...", "objects": [],
"tags": [], "people_hints": [], "pii_flags": [],
"confidence": 0.0, "evidence": "..."}}
No markdown, no prose outside the JSON."""


def llm_cfg(cfg: dict | None) -> dict:
    return (cfg or {}).get("llm") or {}


def enabled(cfg: dict | None) -> bool:
    c = llm_cfg(cfg)
    return bool(c.get("enabled")) and bool(c.get("base_url")) and bool(c.get("asset_id"))


def _extract_json(text: str) -> dict | None:
    """Parse agent output: tolerate fences, {{ }} wrappers, and prose.

    Purple Fabric automation reads JSON from a {{ }}-wrapped payload; the inner
    content may or may not carry its own object braces, so both are attempted.
    """
    if not text or not isinstance(text, str):
        return None
    t = re.sub(r"```(?:json)?", "", text).strip()
    candidates: list[str] = []
    m = re.search(r"\{\{(.*)\}\}", t, re.S)
    if m:
        inner = m.group(1).strip()
        candidates.append(inner)
        candidates.append("{" + inner + "}")
    start, end = t.find("{"), t.rfind("}")
    if start >= 0 and end > start:
        candidates.append(t[start:end + 1])
    for cand in candidates:
        try:
            out = json.loads(cand)
            if isinstance(out, dict):
                return out
        except Exception:
            continue
    return None


_TOKEN_CACHE: dict = {}


def _secret(c: dict, env_key: str, cfg_key: str) -> str:
    """API credentials: env var first (never committed), config value as fallback."""
    return os.environ.get(str(c.get(env_key, "")), "") or str(c.get(cfg_key, "") or "")


def _http(method: str, url: str, headers: dict, payload: dict | None = None,
          timeout: int = 30) -> dict | None:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode() or "{}")
    except (urllib.error.URLError, ValueError, TimeoutError, OSError):
        return None


def get_access_token(cfg: dict | None, force: bool = False) -> str | None:
    """Step 1 of the PF protocol: fetch (and cache) the bearer access token."""
    c = llm_cfg(cfg)
    key = f"{c.get('base_url')}|{c.get('api_key_env')}"
    if not force and _TOKEN_CACHE.get("key") == key and _TOKEN_CACHE.get("token"):
        return _TOKEN_CACHE["token"]
    url = f"{str(c.get('base_url', '')).rstrip('/')}/accesstoken/aubk"
    headers = {"apikey": _secret(c, "api_key_env", "api_key"),
               "username": _secret(c, "username_env", "username"),
               "password": _secret(c, "password_env", "password")}
    body = _http("GET", url, headers, timeout=int(c.get("timeout", 30)))
    token = (body or {}).get("access_token")
    if token:
        _TOKEN_CACHE.update({"key": key, "token": token})
    return token


def invoke_agent(cfg: dict | None, token: str, variables: dict) -> str | None:
    """Step 2: submit the asset run — Input_Text carries the variables as JSON."""
    c = llm_cfg(cfg)
    url = (f"{str(c.get('base_url', '')).rstrip('/')}/magicplatform/v1/"
           f"invokeasset/{c.get('asset_id')}/genai")
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {token}",
               "apikey": _secret(c, "api_key_env", "api_key")}
    payload = {"Input_Text": json.dumps({"task": "describe_asset", **variables},
                                        default=str)}
    body = _http("POST", url, headers, payload, timeout=int(c.get("timeout", 30)))
    return (body or {}).get("trace_id")


def get_result(cfg: dict | None, token: str, trace_id: str) -> dict | None:
    """Step 3: poll the trace until COMPLETED (FAILED/ERROR/timeout -> None)."""
    c = llm_cfg(cfg)
    base = (f"{str(c.get('base_url', '')).rstrip('/')}/magicplatform/v1/"
            f"invokeasset/{c.get('asset_id')}")
    headers = {"Authorization": f"Bearer {token}",
               "apikey": _secret(c, "api_key_env", "api_key")}
    interval = float(c.get("poll_interval", 5))
    deadline = time.time() + float(c.get("poll_timeout", 300))
    while time.time() < deadline:
        body = _http("GET", f"{base}/{trace_id}", headers,
                     timeout=int(c.get("timeout", 30)))
        if body is None:
            return None
        status = str(body.get("status", "")).upper()
        if status == "COMPLETED":
            return body
        if status in ("FAILED", "ERROR"):
            return None
        time.sleep(interval)
    return None


def attach_image(cfg: dict | None, path: str, variables: dict) -> dict:
    """Add image_base64 when llm.send_images is on (10 MB cap). Text-first otherwise."""
    if not llm_cfg(cfg).get("send_images"):
        return variables
    try:
        if os.path.getsize(path) <= 10 * 1024 * 1024:
            with open(path, "rb") as f:
                variables["image_base64"] = base64.b64encode(f.read()).decode()
    except OSError:
        pass
    return variables


def _extract_output(data: dict | None) -> dict | None:
    """Pull the agent's JSON out of the completed-trace envelope (tolerant).

    The completed payload's exact envelope key varies by asset, so several
    candidate keys are tried, then every string value one level deep.
    """
    if not isinstance(data, dict):
        return None
    if isinstance(data.get("caption"), str):
        return data
    found: list[dict] = []

    def _collect(d: dict) -> None:
        if not isinstance(d, dict):
            return
        found.append(d)
        for vv in d.values():   # descend one level for nested envelopes
            if isinstance(vv, dict):
                _collect(vv)

    for k in ("Output_Text", "output", "Output", "result", "response",
              "content", "data", "text", "answer"):
        v = data.get(k)
        if isinstance(v, dict):
            out = _extract_json(json.dumps(v))
            if out:
                _collect(out)
        elif isinstance(v, str):
            out = _extract_json(v)
            if out:
                _collect(out)
    for v in data.values():   # last resort: scan string values one level deep
        if isinstance(v, str) and "{" in v:
            out = _extract_json(v)
            if out:
                _collect(out)
        elif isinstance(v, dict):
            for vv in v.values():
                if isinstance(vv, str) and "{" in vv:
                    out = _extract_json(vv)
                    if out:
                        _collect(out)
    for out in found:   # the agent schema carries a caption — prefer it
        if isinstance(out.get("caption"), str):
            return out
    return found[0] if found else None


def enrich(cfg: dict | None, variables: dict) -> dict | None:
    """Full flow: token -> invoke -> poll -> parse (None on any failure)."""
    if not enabled(cfg):
        return None
    token = get_access_token(cfg)
    if not token:
        return None
    trace = invoke_agent(cfg, token, variables)
    if not trace:   # token may have expired between calls — refresh once, retry
        token = get_access_token(cfg, force=True)
        trace = invoke_agent(cfg, token, variables) if token else None
    if not trace:
        return None
    return _extract_output(get_result(cfg, token, trace))
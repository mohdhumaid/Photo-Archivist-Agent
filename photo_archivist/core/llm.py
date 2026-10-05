"""Purple Fabric LLM client — Automation Digital Expert ("photo-archivist-expert").

Text-first by design: the payload carries OCR/metadata/path only unless
`llm.send_images: true` is set explicitly. Any network/parse failure returns
None and the caller falls back to the mock backend — the pipeline never blocks
on the LLM.

The SYSTEM_PROMPT below must stay in sync with the Perspective (system prompt)
configured on the agent in Purple Fabric Agent Designer — see README.md.
"""
from __future__ import annotations
import base64
import json
import os
import re
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
    return bool(c.get("enabled")) and bool(c.get("invoke_url"))


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


def invoke(cfg: dict | None, variables: dict) -> dict | None:
    """POST the automation input to the Purple Fabric agent; None on any failure."""
    c = llm_cfg(cfg)
    if not enabled(cfg):
        return None
    key = os.environ.get(c.get("api_key_env", "PURPLE_FABRIC_API_KEY"), "")
    payload = {"agent": c.get("agent_id", "photo-archivist-expert"),
               "model": c.get("model", ""),
               "system": SYSTEM_PROMPT,
               "input": variables}
    req = urllib.request.Request(
        c["invoke_url"], data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {key}"} if key else {})},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=int(c.get("timeout", 60))) as r:
            body = json.loads(r.read().decode() or "{}")
    except (urllib.error.URLError, ValueError, TimeoutError, OSError):
        return None
    if isinstance(body, dict):
        text = (body.get("output") or body.get("result") or body.get("response")
                or body.get("content")
                or ((body.get("choices") or [{}])[0].get("message") or {}).get("content")
                or json.dumps(body))
    else:
        text = str(body)
    return _extract_json(text)


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


def enrich(cfg: dict | None, variables: dict) -> dict | None:
    return invoke(cfg, variables)
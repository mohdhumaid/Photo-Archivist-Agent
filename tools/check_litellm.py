"""Standalone LiteLLM / OpenAI-compatible gateway checker (Windows-friendly).

Usage (PowerShell):
  $env:LLM_API_KEY = "sk-..."   # paste the Bearer token your gateway gave you
  python tools/check_litellm.py --base-url https://dev-broccoli-apillmgov.auuat.bank.in/apillmgov/v1/chat/completions --model \"Qwen3 Vision 235b\" [--image path/to/photo.jpg] [--timeout 60]

Exit 0 + parsed caption  => gateway reachable, key accepted, model id valid.
Exit 2 + HTTP 401        => key missing/invalid (most common; see hint below).
Exit 2 + HTTP 404        => model id wrong (ask gateway team for the exact id).
Exit 2 + timeout/DNS     => network or proxy issue, not a code bug.

Never commit a real key: pass it only via $env:LLM_API_KEY (or --api-key once
for a throwaway test; the value is never written anywhere).
"""
from __future__ import annotations
import argparse
import base64
import json
import mimetypes
import os
import sys


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Probe a LiteLLM / OpenAI-compatible chat endpoint.")
    ap.add_argument("--base-url", required=True, help="Full .../v1/chat/completions URL")
    ap.add_argument("--model", required=True, help="Exact model id the gateway expects")
    ap.add_argument("--image", default="", help="Optional photo to attach as base64 image_url")
    ap.add_argument("--api-key", default="", help="Bearer token (prefer $env:LLM_API_KEY)")
    ap.add_argument("--timeout", type=int, default=60)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    key = args.api_key or os.environ.get("LLM_API_KEY", "")
    user_content: list = [{"type": "text",
                           "text": "Reply with exactly this JSON and nothing else: {\"caption\": \"ok\"}"}]
    if args.image:
        mt = mimetypes.guess_type(args.image)[0] or "image/jpeg"
        with open(args.image, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        user_content.append({"type": "image_url",
                             "image_url": {"url": f"data:{mt};base64,{b64}"}})
    payload = {"model": args.model, "temperature": 0.2, "max_tokens": 64,
               "messages": [{"role": "user", "content": user_content}]}
    headers = {"Content-Type": "application/json"}
    masked = ""
    if key:
        headers["Authorization"] = f"Bearer {key}"
        masked = key[:4] + "..." + key[-4:] if len(key) > 8 else "****"
    else:
        print("key: MISSING ($env:LLM_API_KEY unset and --api-key empty)", flush=True)
    print(f"POST {args.base_url}", flush=True)
    print(f"model={args.model} key={masked or 'none'} image={'yes' if args.image else 'no'}", flush=True)
    try:
        import requests
        r = requests.post(args.base_url, json=payload, headers=headers, timeout=args.timeout)
    except Exception as e:
        print(f"RESULT: TRANSPORT-ERROR {type(e).__name__}: {e}")
        print("HINT: DNS/proxy/VPN. On Windows try: $env:HTTPS_PROXY=$env:https_proxy; or ask for the gateway's proxy.")
        return 2
    print(f"RESULT: HTTP {r.status_code}", flush=True)
    body = (r.text or "")[:800]
    print(f"body: {body}", flush=True)
    if r.status_code == 401:
        print("HINT: gateway got NO/WRONG key. Run: $env:LLM_API_KEY=\"sk-...\" then retry. Do not put the key in config.yaml.")
        return 2
    if r.status_code == 404:
        print("HINT: URL or model id wrong. Confirm the exact model string with the gateway team.")
        return 2
    if r.status_code >= 400:
        return 2
    try:
        data = r.json()
        text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content")
        print(f"parsed content: {json.dumps(text)[:300]}")
        print("LITELLM OK - key accepted, model id valid.")
        return 0
    except Exception as e:
        print(f"RESULT: non-JSON 200: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())

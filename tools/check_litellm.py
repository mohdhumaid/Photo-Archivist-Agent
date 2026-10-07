#!/usr/bin/env python3
"""LiteLLM / OpenAI-compatible gateway probe."""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys

import requests


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe a LiteLLM / OpenAI-compatible chat endpoint")
    parser.add_argument("--base-url", required=True, help="Full /v1/chat/completions URL")
    parser.add_argument("--model", required=True, help="Model ID (example: Qwen3 Vision 235b)")
    parser.add_argument("--image", default="", help="Optional image path")
    parser.add_argument("--api-key", default="", help="Bearer token")
    parser.add_argument("--timeout", type=int, default=90, help="Request timeout in seconds")
    return parser


def load_dotenv(path: str = ".env") -> None:
    """Read LLM_API_KEY (key=val) from .env into os.environ, if present."""
    here = os.path.abspath(os.getcwd())
    candidates = [os.path.join(here, path)]
    for _ in range(3):
        here = os.path.dirname(here)
        candidates.append(os.path.join(here, path))
    for cand in candidates:
        if not os.path.isfile(cand):
            continue
        try:
            with open(cand, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, _, v = line.partition("=")
                    k, v = k.strip(), v.strip()
                    if len(v) >= 2 and v[0] == v[-1] and v[0] in ""'":
                        v = v[1:-1]
                    os.environ.setdefault(k, v)
            return
        except OSError:
            continue


def main() -> int:
    load_dotenv()
    args = build_parser().parse_args()

    api_key = args.api_key or os.environ.get("LLM_API_KEY", "")

    if not api_key:
        print("ERROR: API key missing")
        print("Use --api-key, put LLM_API_KEY=<token> in .env, or set the env var")
        return 2

    content = [
        {
            "type": "text",
            "text": "Analyze this image and return valid JSON only. If no image is supplied return {\"status\": \"ok\"}"
        }
    ]

    if args.image:
        if not os.path.exists(args.image):
            print(f"ERROR: Image not found: {args.image}")
            return 2

        mime_type = mimetypes.guess_type(args.image)[0] or "image/jpeg"

        with open(args.image, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{image_b64}"
                }
            }
        )

    payload = {
        "model": args.model,
        "temperature": 0.2,
        "max_tokens": 512,
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ]
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    print("=" * 80)
    print("LiteLLM Gateway Test")
    print("=" * 80)
    print("URL    :", args.base_url)
    print("MODEL  :", args.model)
    print("IMAGE  :", "YES" if args.image else "NO")
    print("=" * 80)

    try:
        response = requests.post(
            args.base_url,
            json=payload,
            headers=headers,
            timeout=args.timeout
        )

    except requests.exceptions.RequestException as e:
        print("\nTRANSPORT ERROR")
        print(str(e))
        return 2

    print("\nHTTP STATUS:", response.status_code)

    try:
        response_json = response.json()
        print("\nRESPONSE JSON:")
        print(json.dumps(response_json, indent=2))
    except Exception:
        print("\nRAW RESPONSE:")
        print(response.text)

    if response.status_code == 200:
        print("\n✅ LiteLLM Gateway Working")
        return 0

    if response.status_code == 401:
        print("\n❌ Invalid API Key")
        return 2

    if response.status_code == 404:
        print("\n❌ Model ID or URL Incorrect")
        return 2

    print("\n❌ Request Failed")
    return 2


if __name__ == "__main__":
    sys.exit(main())

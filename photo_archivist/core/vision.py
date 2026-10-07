"""Step 5: Vision interface - primed with metadata, 1 call/asset. vLLM + Mock."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class VisionResult:
    caption: str = ""
    scene: str = ""
    event_type: str = ""
    objects: list = field(default_factory=list)
    visible_text: str = ""
    face_boxes: list = field(default_factory=list)
    face_embeddings: list = field(default_factory=list)  # one vector per face
    image_vec: list | None = None
    tags: list = field(default_factory=list)
    people_hints: list = field(default_factory=list)  # text-grounded names only
    people_activity: str = ""
    mood: str = ""
    background: str = ""
    location_guess: dict = field(default_factory=dict)
    pii_flags: list = field(default_factory=list)
    confidences: dict = field(default_factory=dict)
    backend: str = "mock"


class VisionBackend:
    def describe(self, path: str, prime: dict) -> VisionResult:
        raise NotImplementedError


class MockVision(VisionBackend):
    """Deterministic stand-in: echoes metadata prime, makes no claims."""

    def describe(self, path: str, prime: dict) -> VisionResult:
        bits = []
        if prime.get("place"):
            bits.append(str(prime["place"]))
        if prime.get("event_hint"):
            bits.append(str(prime["event_hint"]))
        caption = ("; ".join(bits) or "unexamined asset") + " (mock vision - no claim)"
        return VisionResult(caption=caption, backend="mock",
                            confidences={"caption": 0.1})


class VLLMVision(VisionBackend):
    """Generic OpenAI-compatible vision backend: local vLLM *or* LiteLLM gateway.

    Config (config.yaml -> vision_llm): base_url, model, enabled, api_key_env
    (LiteLLM only), temperature, max_tokens, video_mode (keyframe|file_url).
    Mock on any failure - never crashes a scan.
    """

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}

    def describe(self, path: str, prime: dict) -> VisionResult:
        from . import vllm as vllmmod
        out = vllmmod.describe(self.cfg, path, prime)
        if not isinstance(out, dict) or not out.get("caption"):
            return MockVision().describe(path, prime)  # honest fallback
        try:
            conf = float(out.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        conf = min(max(conf, 0.0), 1.0)
        return VisionResult(
            caption=str(out.get("caption") or ""),
            scene=str(out.get("scene") or ""),
            event_type=str(out.get("event_type") or ""),
            objects=[str(x) for x in (out.get("objects") or [])][:50],
            visible_text=str(out.get("visible_text") or ""),
            tags=[str(x) for x in (out.get("tags") or [])][:20],
            people_hints=[str(x) for x in (out.get("people_hints") or [])][:20],
            people_activity=str(out.get("people_activity") or ""),
            mood=str(out.get("mood") or ""),
            background=str(out.get("background") or ""),
            location_guess=dict(out.get("location_guess") or {}),
            pii_flags=[str(x) for x in (out.get("pii_flags") or [])][:20],
            confidences={"caption": conf, "llm": conf},
            backend="vllm")


# Legacy alias: "llm" was the Purple Fabric backend name. It now resolves to
# the vLLM/LiteLLM gateway (or mock when unconfigured) so stale configs keep
# working without the removed purple-fabric client.
LLMVision = VLLMVision


def get_backend(name: str, cfg: dict | None = None) -> VisionBackend:
    if name == "mock":
        return MockVision()
    if name in ("llm", "vllm", "local"):
        from . import vllm as vllmmod
        if vllmmod.enabled(cfg):
            return VLLMVision(cfg)
        return MockVision()   # not configured -> mock, never a crash
    # unknown backend name -> mock so a stale config never crashes a scan.
    return MockVision()

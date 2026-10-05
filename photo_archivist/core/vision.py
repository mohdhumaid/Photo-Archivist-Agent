"""Step 5: Vision interface — primed with metadata, 1 call/asset. Local + Mock."""
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
    face_embeddings: list = field(default_factory=list)  # one 512-d vector per face
    image_vec: list | None = None                        # CLIP embedding, when available
    tags: list = field(default_factory=list)
    people_hints: list = field(default_factory=list)  # text-grounded names from the LLM
    pii_flags: list = field(default_factory=list)     # LLM-detected identifiers
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
        caption = ("; ".join(bits) or "unexamined asset") + " (mock vision — no claim)"
        return VisionResult(caption=caption, backend="mock",
                            confidences={"caption": 0.1})


class LLMVision(VisionBackend):
    """Purple Fabric Automation Digital Expert — returns mock on any failure."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}

    def describe(self, path: str, prime: dict) -> VisionResult:
        import os
        from . import llm as llmmod
        variables = {
            "file_name": prime.get("file_name") or os.path.basename(path),
            "file_type": prime.get("file_type") or "",
            "mime": prime.get("mime") or "",
            "path_segments": prime.get("path_segments") or [],
            "ocr_text": prime.get("ocr_text") or "",
            "metadata_json": prime.get("metadata") or {},
        }
        llmmod.attach_image(self.cfg, path, variables)
        out = llmmod.enrich(self.cfg, variables)
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
            tags=[str(x) for x in (out.get("tags") or [])][:20],
            people_hints=[str(x) for x in (out.get("people_hints") or [])][:20],
            pii_flags=[str(x) for x in (out.get("pii_flags") or [])][:20],
            confidences={"caption": conf, "llm": conf},
            backend="purple_fabric")


def get_backend(name: str, cfg: dict | None = None) -> VisionBackend:
    if name == "mock":
        return MockVision()
    if name == "llm":
        from . import llm as llmmod
        if llmmod.enabled(cfg):
            return LLMVision(cfg)
        return MockVision()   # not configured -> mock, never a crash
    # "local" backends are unavailable under org policy (no model downloads);
    # route to mock so a stale config never crashes a scan.
    return MockVision()

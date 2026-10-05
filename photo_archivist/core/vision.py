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
    tags: list = field(default_factory=list)
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


def get_backend(name: str) -> VisionBackend:
    if name == "mock":
        return MockVision()
    # local backends (Moondream/LLaVA) plug in here; default to mock when unavailable
    try:
        from .vision_local import LocalVision  # type: ignore
        return LocalVision()
    except Exception:
        return MockVision()

"""Purple Fabric LLM integration tests: config gating, output parsing,
backend fallback, prime construction, and result merging."""
from photo_archivist.core import llm as llmmod
from photo_archivist.core import vision as visionmod


def test_llm_disabled_by_default():
    assert llmmod.enabled({}) is False
    assert llmmod.enabled({"llm": {"enabled": True, "invoke_url": ""}}) is False
    assert llmmod.enabled({"llm": {"enabled": False, "invoke_url": "https://x"}}) is False
    assert llmmod.enabled({"llm": {"enabled": True, "invoke_url": "https://x"}}) is True


def test_get_backend_llm_falls_back_to_mock_when_unconfigured():
    assert isinstance(visionmod.get_backend("llm", {}), visionmod.MockVision)
    assert isinstance(visionmod.get_backend("llm", None), visionmod.MockVision)


def test_extract_json_plain_and_wrapped():
    assert llmmod._extract_json('{"caption": "c"}') == {"caption": "c"}
    assert llmmod._extract_json('{{"caption": "c"}}') == {"caption": "c"}
    fenced = '```json\n{"caption": "c"}\n```'
    assert llmmod._extract_json(fenced) == {"caption": "c"}
    with_prose = 'Here is the result: {"caption": "c"} hope it helps'
    assert llmmod._extract_json(with_prose) == {"caption": "c"}
    assert llmmod._extract_json("no json here") is None
    assert llmmod._extract_json("") is None


def test_invoke_unreachable_endpoint_returns_none():
    cfg = {"llm": {"enabled": True, "invoke_url": "http://127.0.0.1:1/invoke",
                   "timeout": 1}}
    assert llmmod.invoke(cfg, {"file_name": "x"}) is None
    assert llmmod.enrich(cfg, {"file_name": "x"}) is None


def test_llmvision_parses_agent_output(monkeypatch):
    cfg = {"llm": {"enabled": True, "invoke_url": "https://pf.example/invoke"}}
    monkeypatch.setattr(
        llmmod, "enrich", lambda c, v: {
            "caption": "Branch launch at Indiranagar",
            "scene": "office event", "event_type": "branch_launch",
            "objects": ["banner", "people"], "tags": ["branch launch", "indiranagar"],
            "people_hints": ["Anita Rao (from ocr_text)"],
            "pii_flags": ["phone_number"], "confidence": 0.85,
            "evidence": "ocr_text mentions branch opening"})
    vb = visionmod.LLMVision(cfg)
    vr = vb.describe("/tmp/anything.jpg", {"file_name": "anything.jpg"})
    assert vr.backend == "purple_fabric"
    assert vr.confidences["caption"] == 0.85
    assert "branch launch" in vr.tags and vr.people_hints and vr.pii_flags


def test_llmvision_falls_back_to_mock_on_failure(monkeypatch):
    cfg = {"llm": {"enabled": True, "invoke_url": "https://pf.example/invoke"}}
    monkeypatch.setattr(llmmod, "enrich", lambda c, v: None)
    vr = visionmod.LLMVision(cfg).describe("/tmp/a.jpg", {"place": "Indiranagar"})
    assert vr.backend == "mock" and "no claim" in vr.caption


def test_attach_image_respects_privacy_flag(tmp_path):
    img = tmp_path / "a.jpg"
    img.write_bytes(b"\xff\xd8\xff" + b"0" * 100)
    # default: text-first, no image attached
    out = llmmod.attach_image({"llm": {"send_images": False}}, str(img), {})
    assert "image_base64" not in out
    out = llmmod.attach_image({"llm": {"send_images": True}}, str(img), {})
    assert "image_base64" in out


def test_pipeline_prime_contains_llm_context(tmp_path):
    from docx import Document
    from photo_archivist.core import pipeline as pipe
    from photo_archivist.core.vision import VisionResult

    src = tmp_path / "Indiranagar" / "Branch-Launches"
    src.mkdir(parents=True)
    p = src / "loan_letter.docx"
    doc = Document()
    doc.add_paragraph("sanction letter")
    doc.save(str(p))

    captured = {}

    class CaptureVB:
        def describe(self, path, prime):
            captured.update(prime)
            return VisionResult(backend="capture")

    cfg = {"embeddings": {"backend": "hash"}}
    rec = pipe.process_file(str(p), cfg, vision_backend=CaptureVB())
    assert captured["file_name"] == "loan_letter.docx"
    assert captured["file_type"] == "document"
    assert "Branch-Launches" in captured["path_segments"]
    assert isinstance(captured["metadata"], dict)
    assert rec["vectors"]["image"] is None

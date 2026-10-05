"""Purple Fabric Magic Platform tests: gating, token caching, invoke/poll
flow, output extraction, and graceful fallbacks (no network in tests)."""
from photo_archivist.core import llm as llmmod
from photo_archivist.core import vision as visionmod

PF_CFG = {"llm": {"enabled": True, "base_url": "https://pf.example",
                  "asset_id": "asset-1", "api_key": "k", "username": "u",
                  "password": "p", "poll_interval": 0.001, "poll_timeout": 5}}

AGENT_OUT = {"caption": "Branch launch at Indiranagar", "scene": "office",
             "event_type": "branch_launch", "objects": ["banner"],
             "tags": ["branch launch"], "people_hints": [], "pii_flags": [],
             "confidence": 0.85, "evidence": "ocr"}


def test_llm_gating():
    assert llmmod.enabled({}) is False
    assert llmmod.enabled({"llm": {"enabled": True}}) is False          # no url/asset
    assert llmmod.enabled({"llm": {"enabled": True, "base_url": "x"}}) is False
    assert llmmod.enabled(PF_CFG) is True
    assert isinstance(visionmod.get_backend("llm", {}), visionmod.MockVision)
    assert isinstance(visionmod.get_backend("llm", None), visionmod.MockVision)


def _fake_http_factory():
    calls = []

    def fake(method, url, headers, payload=None, timeout=30):
        calls.append((method, url, headers, payload))
        if url.endswith("/accesstoken/aubk"):
            assert headers["apikey"] == "k"
            assert headers["username"] == "u" and headers["password"] == "p"
            return {"access_token": "tok-1"}
        if url.endswith("/genai"):
            assert headers["Authorization"] == "Bearer tok-1"
            import json
            vars_sent = json.loads(payload["Input_Text"])
            assert vars_sent["task"] == "describe_asset"
            assert vars_sent["file_name"] == "loan.docx"
            return {"trace_id": "trace-9"}
        if url.endswith("/trace-9"):
            import json
            return {"status": "COMPLETED", "Output_Text": json.dumps(AGENT_OUT)}
        raise AssertionError(f"unexpected {method} {url}")

    fake.calls = calls
    return fake


def test_full_flow_token_invoke_poll(monkeypatch):
    llmmod._TOKEN_CACHE.clear()
    monkeypatch.setattr(llmmod, "_http", _fake_http_factory())
    out = llmmod.enrich(PF_CFG, {"file_name": "loan.docx", "ocr_text": ""})
    assert out == AGENT_OUT


def test_token_cached_across_calls(monkeypatch):
    llmmod._TOKEN_CACHE.clear()
    fake = _fake_http_factory()
    monkeypatch.setattr(llmmod, "_http", fake)
    cfg = dict(PF_CFG)
    assert llmmod.get_access_token(cfg) == "tok-1"
    assert llmmod.get_access_token(cfg) == "tok-1"
    token_calls = [c for c in fake.calls if c[0] == "GET" and "accesstoken" in c[1]]
    assert len(token_calls) == 1   # second call served from cache


def test_failed_status_returns_none(monkeypatch):
    llmmod._TOKEN_CACHE.clear()

    def fake(method, url, headers, payload=None, timeout=30):
        if url.endswith("/accesstoken/aubk"):
            return {"access_token": "t"}
        if url.endswith("/genai"):
            return {"trace_id": "tx"}
        return {"status": "FAILED"}

    monkeypatch.setattr(llmmod, "_http", fake)
    assert llmmod.enrich(PF_CFG, {"file_name": "x"}) is None


def test_extract_json_plain_and_wrapped():
    assert llmmod._extract_json('{"caption": "c"}') == {"caption": "c"}
    assert llmmod._extract_json('{{"caption": "c"}}') == {"caption": "c"}
    fenced = '```json\n{"caption": "c"}\n```'
    assert llmmod._extract_json(fenced) == {"caption": "c"}
    with_prose = 'Here is the result: {"caption": "c"} hope it helps'
    assert llmmod._extract_json(with_prose) == {"caption": "c"}
    assert llmmod._extract_json("no json here") is None
    assert llmmod._extract_json("") is None


def test_llmvision_uses_backend_on_success(monkeypatch):
    cfg = {"llm": {"enabled": True, "base_url": "x", "asset_id": "y"}}
    monkeypatch.setattr(llmmod, "enrich", lambda c, v: dict(AGENT_OUT))
    vr = visionmod.LLMVision(cfg).describe("/tmp/anything.jpg",
                                           {"file_name": "anything.jpg"})
    assert vr.backend == "purple_fabric"
    assert vr.confidences["caption"] == 0.85
    assert "branch launch" in vr.tags


def test_llmvision_falls_back_to_mock_on_failure(monkeypatch):
    cfg = {"llm": {"enabled": True, "base_url": "x", "asset_id": "y"}}
    monkeypatch.setattr(llmmod, "enrich", lambda c, v: None)
    vr = visionmod.LLMVision(cfg).describe("/tmp/a.jpg", {"place": "Indiranagar"})
    assert vr.backend == "mock" and "no claim" in vr.caption


def test_attach_image_respects_privacy_flag(tmp_path):
    img = tmp_path / "a.jpg"
    img.write_bytes(b"\xff\xd8\xff" + b"0" * 100)
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


def test_network_failure_returns_none(monkeypatch):
    monkeypatch.setattr(llmmod, "_http", lambda *a, **k: None)
    assert llmmod.enrich(PF_CFG, {"file_name": "x"}) is None


def test_extract_output_variants():
    import json
    assert llmmod._extract_output({"caption": "c"}) == {"caption": "c"}
    assert llmmod._extract_output({"Output_Text": json.dumps({"caption": "c"})}) == {"caption": "c"}
    assert llmmod._extract_output(
        {"result": {"nested": {"caption": "c"}}}) == {"caption": "c"}
    assert llmmod._extract_output({"status": "COMPLETED"}) is None
    assert llmmod._extract_output(None) is None
    assert llmmod._extract_output(
        {"status": "COMPLETED", "Output_Text": '{{"caption": "c"}}'}) == {"caption": "c"}

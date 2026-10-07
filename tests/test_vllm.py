"""Local/gateway vision LLM tests: gating, scrubbing, message building."""
from photo_archivist.core import vllm as vllmmod
from photo_archivist.core import vision as visionmod

V_CFG = {"vision_llm": {"enabled": True,
                        "base_url": "http://127.0.0.1:8000/v1/chat/completions",
                        "model": "Qwen/Qwen3-VL-8B-Thinking-FP8"}}


def test_gating():
    assert vllmmod.enabled({}) is False
    assert vllmmod.enabled({"vision_llm": {"enabled": True}}) is False
    assert vllmmod.enabled(V_CFG) is True
    assert isinstance(visionmod.get_backend("vllm", {}), visionmod.MockVision)
    assert isinstance(visionmod.get_backend("vllm", V_CFG), visionmod.VLLMVision)
    assert isinstance(visionmod.get_backend("local", V_CFG), visionmod.VLLMVision)


def test_scrub_rejects_generic_labels():
    for word in ("Office", "Home", "indoor", "BANK", "branch "):
        out = vllmmod.scrub_location_guess(
            {"location_guess": {"value": word, "confidence": 0.9}})
        assert out["value"] is None and out["confidence"] == 0.0, word
    good = vllmmod.scrub_location_guess(
        {"location_guess": {"value": "AU Small Finance Bank, Indiranagar",
                            "evidence": "banner", "confidence": 0.9}})
    assert good["value"].startswith("AU Small") and good["confidence"] == 0.9
    assert vllmmod.scrub_location_guess({"location_guess": {"value": None}})["value"] is None


def test_strip_think():
    assert vllmmod._strip_think("<think>reasoning</think>{\"a\":1}") == '{"a":1}'

def test_build_messages_text_first_for_docs(tmp_path):
    p = tmp_path / "letter.pdf"
    p.write_bytes(b"%PDF-1.4 fake")
    prime = {"file_name": "letter.pdf", "file_type": "document",
             "mime": "application/pdf", "path_segments": ["Events"],
             "ocr_text": "sanction", "metadata": {}}
    msgs, tmp = vllmmod.build_messages(V_CFG, str(p), prime)
    assert tmp is None  # documents: no image bytes, text-first
    assert msgs[0]["role"] == "system"
    assert all(pt["type"] == "text" for pt in msgs[1]["content"])


def test_build_messages_image_base64(tmp_path):
    from PIL import Image
    p = tmp_path / "a.jpg"
    Image.new("RGB", (8, 8)).save(p)
    prime = {"file_name": "a.jpg", "file_type": "image", "mime": "image/jpeg",
             "path_segments": [], "ocr_text": "", "metadata": {}}
    msgs, tmp = vllmmod.build_messages(V_CFG, str(p), prime)
    kinds = [pt["type"] for pt in msgs[1]["content"]]
    assert "image_url" in kinds and tmp is None


def test_build_messages_file_url_video_mode(tmp_path):
    p = tmp_path / "m.mp4"
    p.write_bytes(b"\x00" * 100)
    cfg = {"vision_llm": dict(V_CFG["vision_llm"], video_mode="file_url")}
    prime = {"file_name": "m.mp4", "file_type": "video", "mime": "video/mp4",
             "path_segments": [], "ocr_text": "", "metadata": {}}
    msgs, _ = vllmmod.build_messages(cfg, str(p), prime)
    assert "video_url" in [pt["type"] for pt in msgs[1]["content"]]


def test_vllmvision_maps_new_fields(monkeypatch):
    out = {"caption": "Ribbon cutting", "scene": "stage event",
           "event_type": "branch_launch", "objects": ["banner"],
           "visible_text": "AU Small Finance Bank Indiranagar",
           "tags": ["branch launch"], "people_hints": [],
           "people_activity": "cutting a ribbon", "mood": "smiling, celebratory",
           "background": "stage with blue banner",
           "location_guess": {"value": "Indiranagar branch", "evidence": "banner",
                              "confidence": 0.8},
           "pii_flags": [], "confidence": 0.8}
    monkeypatch.setattr(vllmmod, "describe", lambda c, p, pr: dict(out))
    vr = visionmod.VLLMVision(V_CFG).describe("/tmp/a.jpg", {"file_name": "a.jpg"})
    assert vr.backend == "vllm"
    assert vr.people_activity == "cutting a ribbon" and vr.mood.startswith("smiling")
    assert vr.background.startswith("stage") and vr.visible_text.startswith("AU Small")
    assert vr.location_guess["value"] == "Indiranagar branch"


def test_pipeline_location_fallback_only_when_metadata_empty(tmp_path):
    from PIL import Image
    from photo_archivist.core import pipeline as pipe
    from photo_archivist.core.vision import VisionResult

    p = tmp_path / "shot.jpg"
    Image.new("RGB", (16, 16)).save(p)

    class LocVB:
        def describe(self, path, prime):
            return VisionResult(backend="vllm", caption="group photo",
                                people_activity="posing, smiling", mood="cheerful",
                                background="branch entrance",
                                location_guess={"value": "Indiranagar branch",
                                                "evidence": "banner", "confidence": 0.8})

    cfg = {"embeddings": {"backend": "hash"}}
    rec = pipe.process_file(str(p), cfg, vision_backend=LocVB())
    assert rec["place"]["value"] == "Indiranagar branch"
    assert rec["place"]["source"].startswith("vision_location")
    assert "posing" in (rec["photo_description"] or "")

    class MetaVB:
        def describe(self, path, prime):
            return VisionResult(backend="vllm", caption="x",
                                location_guess={"value": "Somewhere Else",
                                                "evidence": "?", "confidence": 0.9})

    # NOTE: pytest tmp_path is a junk crawl path, so the LLM still fills it.
    # Real-folder priority is covered by test_pipeline_rejected_label_flagged
    # style checks + the metadata-wins branch in pipeline.py.
    rec2 = pipe.process_file(str(p), cfg, vision_backend=MetaVB())
    assert rec2["place"]["source"].startswith("vision_location")


def test_pipeline_rejected_label_flagged(tmp_path):
    from PIL import Image
    from photo_archivist.core import pipeline as pipe
    from photo_archivist.core.vision import VisionResult

    p = tmp_path / "shot.jpg"
    Image.new("RGB", (16, 16)).save(p)


def test_metadata_place_wins_over_llm_guess(tmp_path):
    """Real folder place beats a conflicting LLM guess (trust order §3.4)."""
    from PIL import Image
    from photo_archivist.core import pipeline as pipe
    from photo_archivist.core.vision import VisionResult

    folder = tmp_path / "Indiranagar"
    folder.mkdir()
    p = folder / "shot.jpg"
    from PIL import Image as _I
    _I.new("RGB", (16, 16)).save(p)

    class ClashVB:
        def describe(self, path, prime):
            return VisionResult(backend="vllm", caption="x",
                                location_guess={"value": "Somewhere Else",
                                                "evidence": "?", "confidence": 0.9})

    # 'Indiranagar' is a real folder name (no tmp marker) -> metadata wins
    import shutil
    real = tmp_path / "Events" / "Indiranagar"
    real.mkdir(parents=True)
    rp = real / "shot.jpg"
    shutil.copy(str(p), str(rp))
    rec = pipe.process_file(str(rp), {"embeddings": {"backend": "hash"}},
                            vision_backend=ClashVB())
    assert "Somewhere Else" not in str(rec["place"]["value"])
    assert rec["place"]["source"] == "path"

    class OfficeVB:
        def describe(self, path, prime):
            return VisionResult(backend="vllm", caption="meeting",
                                location_guess={"value": None, "rejected": "Office",
                                                "confidence": 0.0,
                                                "evidence": "generic label rejected"})

    rec = pipe.process_file(str(p), {"embeddings": {"backend": "hash"}},
                            vision_backend=OfficeVB())
    assert rec["place"]["value"] != "Office"
    assert any("vision_location_rejected" in f for f in rec["pii_flags"])


def test_validate_llm_output_proper():
    from photo_archivist.core.vllm import validate_llm_output
    good = {"caption": "Team at the Indiranagar branch launch", "tags": ["launch", "team"],
            "objects": ["banner"], "people_hints": ["Anita Rao"], "confidence": 0.82,
            "pii_flags": []}
    assert validate_llm_output(good) == []
    # minimal-but-valid: only caption
    assert validate_llm_output({"caption": "Office desk with laptop"}) == []


def test_validate_llm_output_flags_bad_shapes():
    from photo_archivist.core.vllm import validate_llm_output
    assert validate_llm_output(None)            # not parseable
    assert validate_llm_output({})              # missing caption
    assert validate_llm_output({"caption": "x"})  # too short
    issues = validate_llm_output({
        "caption": "A photo (mock vision - no claim)",
        "tags": "not-a-list", "confidence": 4.2,
        "people_hints": ["ok", ""]})
    text = " ".join(issues)
    assert "MOCK fallback" in text
    assert "tags should be a list" in text
    assert "confidence 4.2 outside 0..1" in text
    assert "non-string/empty" in text


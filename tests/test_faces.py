"""OpenCV Haar face detection tests (skip gracefully when opencv absent)."""
import pytest
from photo_archivist.core import faces as facemod
from photo_archivist.core import report as reportmod


def test_detect_faces_never_raises(tmp_path):
    # missing file / non-image -> [] regardless of whether opencv is installed
    assert facemod.detect_faces(str(tmp_path / "nope.jpg")) == []
    bogus = tmp_path / "fake.jpg"
    bogus.write_text("not an image")
    assert facemod.detect_faces(str(bogus)) == []


def test_available_reports_tuple():
    ok, msg = facemod.available()
    assert isinstance(ok, bool) and isinstance(msg, str) and msg


def test_cascade_path_is_inside_wheel_when_installed():
    p = facemod.cascade_path()
    if p is None:
        pytest.skip("opencv not installed on this machine")
    assert p.endswith(facemod.CASCADE_NAME)
    import os
    assert os.path.exists(p)   # shipped inside the wheel — no runtime download


def test_detect_blank_image_returns_list():
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "blank.png")
        cv2.imwrite(p, np.zeros((120, 120), dtype=np.uint8))
        out = facemod.detect_faces(p)
    assert isinstance(out, list) and out == []   # blank image -> no faces


def test_report_faces_coverage_and_unnamed_question():
    faces3 = [{"box": [1, 2, 30, 30], "detector": "haar_frontal_default"}] * 3
    rec = [{"source_path": "/s/a.jpg", "type": "image", "face_boxes": faces3,
            "people": [], "tags": [], "pii_flags": [],
            "taken_at_confidence": 0.9, "raw_metadata": {"exif": {}},
            "place": {}, "metadata_status": "partial"}]
    rep = reportmod.build_report(rec, [{"action": "new_folder", "score": 0.0}],
                                 unnamed_face_ask_n=3)
    assert rep["coverage"]["faces_pct"] == 100.0
    assert any("UNNAMED-FACE" in q and "3 faces" in q for q in rep["questions"])
    # below threshold -> no question
    rep2 = reportmod.build_report(rec, [{"action": "new_folder", "score": 0.0}],
                                  unnamed_face_ask_n=5)
    assert not any("UNNAMED-FACE" in q for q in rep2["questions"])
    # named people present -> no question
    named = [dict(rec[0], people=[{"name": "X"}])]
    rep3 = reportmod.build_report(named, [{"action": "new_folder", "score": 0.0}],
                                  unnamed_face_ask_n=3)
    assert not any("UNNAMED-FACE" in q for q in rep3["questions"])


def test_pipeline_record_includes_face_boxes(tmp_path):
    from docx import Document
    from photo_archivist.core import pipeline as pipe
    # document type -> face detection not even attempted -> key exists, empty
    p = tmp_path / "doc.docx"
    doc = Document()
    doc.add_paragraph("hello")
    doc.save(str(p))
    rec = pipe.process_file(str(p), {"embeddings": {"backend": "hash"}})
    assert rec["face_boxes"] == []
"""OpenCV Haar face detection tests (skip gracefully when opencv absent)."""
import os
import pytest
from photo_archivist.core import faces as facemod
from photo_archivist.core import report as reportmod
from photo_archivist.core import people as peoplemod


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
    assert any("UNNAMED-FACE" in q and "3 of 3 faces unnamed" in q for q in rep["questions"])
    # below threshold -> no question
    rep2 = reportmod.build_report(rec, [{"action": "new_folder", "score": 0.0}],
                                  unnamed_face_ask_n=5)
    assert not any("UNNAMED-FACE" in q for q in rep2["questions"])
    # named people present -> no question
    named = [dict(rec[0], people=[{"name": "A"}, {"name": "B"}, {"name": "C"}])]
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


def test_clean_name_normalisation():
    assert facemod.clean_name("AkhilVerma.jpeg") == "Akhil Verma"
    assert facemod.clean_name("Anmol-Padhye.png") == "Anmol Padhye"
    assert facemod.clean_name("sanjay agarwal.png") == "Sanjay Agarwal"
    assert facemod.clean_name("uttamtibrewal.png") == "Uttam Tibrewal"
    assert facemod.clean_name("Yogesh-Jain.png") == "Yogesh Jain"
    assert facemod.clean_name("Yogesh-Soni.png") == "Yogesh Soni"


def test_enroll_folder_and_match(tmp_path):
    import json
    from photo_archivist.core import people as peoplemod
    lib_path = str(tmp_path / "test_faces.json")
    results = facemod.enroll_folder("faces", lib_path=lib_path)
    if not results:
        pytest.skip("faces/ folder not present or empty")
    assert os.path.exists(lib_path)
    with open(lib_path) as f:
        data = json.load(f)
    assert "Sanjay Agarwal" in data
    assert "Uttam Tibrewal" in data
    assert len(data["Sanjay Agarwal"]) in (128, facemod.EMBED_SIZE * facemod.EMBED_SIZE)
    # Self-match should score 1.00
    m = peoplemod.match_local(data["Sanjay Agarwal"], data, threshold=0.95)
    assert m is not None
    assert m["name"] == "Sanjay Agarwal"
    assert m["confidence"] == 1.0


def test_persons_for_faces_tags_every_face():
    """3 detections -> 3 person entries (region + match + placeholder)."""
    from photo_archivist.core import people as peoplemod
    lib = {}
    raw = {"XMP-mwg-rs:RegionInfo": [{"Name": "Anita Rao",
        "Rectangle": {"X": 0.05, "Y": 0.05, "W": 0.20, "H": 0.20}}]}
    dets = [{"box": [0.05, 0.05, 0.20, 0.20], "embedding": None},
            {"box": [0.40, 0.05, 0.20, 0.20], "embedding": None},
            {"box": [0.70, 0.05, 0.20, 0.20], "embedding": None}]
    persons, unnamed = peoplemod.persons_for_faces(
        dets, raw, lib, threshold=0.99,
        people_hints=["Vikram Shah (banner)"])
    assert len(persons) == 3
    assert persons[0]["name"] == "Anita Rao" and persons[0]["confidence"] == 0.99
    assert persons[1]["name"] == "Vikram Shah"
    assert persons[0]["box"] and persons[1]["box"] and persons[2]["box"]
    assert unnamed == 1 and persons[2]["name"].startswith("Unknown Person")


def test_persons_for_faces_empty_detections_but_regions_kept():
    """No detections -> pipeline keeps region names (tested separately)."""
    from photo_archivist.core import people as peoplemod
    persons, unnamed = peoplemod.persons_for_faces([], {}, {}, threshold=0.6)
    assert persons == [] and unnamed == 0

def test_filename_hint_enrolled_and_unenrolled():
    """'sanjay agarwal.png' names the face from the filename."""
    det = [{"box": [10, 10, 50, 50], "embedding": [0.0] * 128}]
    # enrolled name matches filename -> 0.60, face_filename
    persons, unnamed = peoplemod.persons_for_faces(
        det, {}, {"Sanjay Agarwal": [0.0] * 128}, threshold=0.99,
        path=r"D:\AI Project\faces\sanjay agarwal.png")
    assert persons[0]["name"] == "Sanjay Agarwal"
    assert persons[0]["source"] == "face_filename"
    assert persons[0]["confidence"] == 0.60
    assert unnamed == 0
    # plausible but not enrolled -> still named, lower confidence
    persons, _ = peoplemod.persons_for_faces(
        [{"box": [1, 1, 9, 9], "embedding": None}], {}, {}, threshold=0.9,
        path="photos/anita rao.png")
    assert persons[0]["name"] == "Anita Rao"
    assert persons[0]["confidence"] == 0.45


def test_filename_hint_rejects_junk_and_uses_folder_only_if_enrolled():
    det = [{"box": [2, 2, 8, 8], "embedding": None}]
    # digits / IMG- pattern never becomes a person
    persons, unnamed = peoplemod.persons_for_faces(
        det, {}, {}, threshold=0.9, path="D:/DCIM/IMG-20250714-WA0012.jpg")
    assert persons[0]["source"] == "face_unmatched"
    assert unnamed == 1
    # folder name trusted only when it matches an enrolled name
    persons, _ = peoplemod.persons_for_faces(
        det, {}, {"Sanjay Agarwal": [0.0] * 128}, threshold=0.99,
        path="C:/Albums/Sanjay Agarwal/IMG_0001.jpg")
    assert persons[0]["name"] == "Sanjay Agarwal"
    assert persons[0]["source"] == "face_folder"
    # non-enrolled folder (event name) is never a person
    persons, _ = peoplemod.persons_for_faces(
        det, {}, {}, threshold=0.9, path="C:/Albums/New folder/IMG_0001.jpg")
    assert persons[0]["source"] == "face_unmatched"

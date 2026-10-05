"""Search tests: filtering, prefix match, year filter, provenance reason."""
from photo_archivist.core import index as idx
from photo_archivist.core import search as searchmod


def _seed(tmp_path):
    db = str(tmp_path / "i.db")
    c = idx.connect(db)
    recs = [
        {"file_id": "f1", "sha256": "a" * 64, "source_path": "/src/AkhilVerma.jpeg",
         "organised_path": "Organised/Inbox/AkhilVerma.jpeg", "type": "image",
         "caption": "Cline / Images (mock vision — no claim)", "tags": ["Cline", "Images"],
         "ocr_text": "Loan sanction letter 2023", "place": {"value": "Cline / Images",
         "source": "path"}, "event": {"value": "Documents"}, "taken_at": "2023-05-01T10:00:00",
         "taken_at_source": "exif", "taken_at_confidence": 0.99, "people": []},
        {"file_id": "f2", "sha256": "b" * 64, "source_path": "/src/Yogesh-Jain.png",
         "organised_path": "Organised/Inbox/Yogesh-Jain.png", "type": "image",
         "caption": "Cline / Images (mock vision — no claim)", "tags": ["Cline", "Images"],
         "ocr_text": "", "place": {"value": "Cline / Images", "source": "path"},
         "event": {"value": "Documents"}, "taken_at": "2024-01-01T00:00:00",
         "taken_at_source": "fs_birthtime", "taken_at_confidence": 0.3, "people": []},
    ]
    for r in recs:
        idx.upsert_file(c, r)
    c.close()
    return db


def test_search_filters_to_matching_rows(tmp_path):
    db = _seed(tmp_path)
    hits = searchmod.search(db, "Akhil")
    assert len(hits) == 1 and "AkhilVerma" in hits[0]["source_path"]


def test_search_nonexistent_returns_empty(tmp_path):
    db = _seed(tmp_path)
    assert searchmod.search(db, "Indiranagar branch launch") == []


def test_search_year_filter(tmp_path):
    db = _seed(tmp_path)
    hits = searchmod.search(db, "sanction letter 2023")
    assert len(hits) == 1 and hits[0]["file_id"] == "f1"
    hits2 = searchmod.search(db, "Cline 2024")
    assert [h["file_id"] for h in hits2] == ["f2"]


def test_search_reason_has_provenance(tmp_path):
    db = _seed(tmp_path)
    hits = searchmod.search(db, "Akhil")
    assert "from path" in hits[0]["reason"] and "from exif" in hits[0]["reason"]
    assert "text match: Akhil" in hits[0]["reason"]

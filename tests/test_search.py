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


def test_role_match_is_whole_word_only():
    # documented behaviour: "MD letter" resolves MD -> person name
    f = searchmod.parse_query("MD signed letter", {"MD": "Anita Rao"})
    assert f["person"] == "Anita Rao"
    # substring traps: role must not fire inside another word; empty values ignored
    f2 = searchmod.parse_query("bmw photos", {"BM": "Nobody"})
    assert not f2["person"]
    f3 = searchmod.parse_query("cmd tools", {"MD": "Anita Rao"})
    assert not f3["person"]
    f4 = searchmod.parse_query("anything", {"MD": ""})
    assert not f4["person"]


def test_people_map_title_annotates_reason(tmp_path):
    db = str(tmp_path / "i.db")
    c = idx.connect(db)
    rec = {"file_id": "p1", "sha256": "c" * 64,
           "source_path": "/src/Anita-Rao.png", "type": "image",
           "caption": "office photo", "tags": ["office"],
           "ocr_text": "letter signed by Anita Rao",
           "place": {"value": "Indiranagar", "source": "path"},
           "event": {"value": None}, "taken_at": None, "taken_at_source": None,
           "taken_at_confidence": 0.3,
           "people": [{"name": "Anita Rao", "source": "xmp_mwg_region",
                       "confidence": 0.99}]}
    idx.upsert_file(c, rec)
    c.close()
    hits = searchmod.search(db, "Anita",
                            people_map={"Anita Rao": "Managing Director"})
    assert len(hits) == 1
    assert "Anita Rao [xmp_mwg_region]" in hits[0]["reason"]
    assert "Managing Director" in hits[0]["reason"]


def _seed_people(tmp_path):
    """Two similar names + keyword-noise rows for relevance tests."""
    from photo_archivist.core.embed import text_vector
    db = str(tmp_path / "r.db")
    c = idx.connect(db)
    recs = [
        {"file_id": "yj", "sha256": "1" * 64, "type": "image",
         "source_path": "/src/Yogesh-Jain.png",
         "organised_path": "Organised/2024/Yogesh-Jain.png",
         "caption": "Yogesh Jain at the annual review", "tags": ["review"],
         "ocr_text": "", "taken_at": "2024-05-01T00:00:00",
         "taken_at_source": "exif", "taken_at_confidence": 0.9,
         "people": [{"name": "Yogesh Jain", "source": "face_match_local",
                     "confidence": 0.87}],
         "vectors": {"text": text_vector("yogesh jain annual review")}},
        {"file_id": "ys", "sha256": "2" * 64, "type": "image",
         "source_path": "/src/Yogesh-Soni.png",
         "organised_path": "Organised/2024/Yogesh-Soni.png",
         "caption": "Yogesh Soni at the town hall", "tags": ["town hall"],
         "ocr_text": "", "taken_at": "2024-06-01T00:00:00",
         "taken_at_source": "exif", "taken_at_confidence": 0.9,
         "people": [{"name": "Yogesh Soni", "source": "face_match_local",
                     "confidence": 0.85}],
         "vectors": {"text": text_vector("yogesh soni town hall")}},
        {"file_id": "sanjay", "sha256": "3" * 64, "type": "image",
         "source_path": "/src/sanjay agarwal.png",
         "organised_path": "Organised/2024/sanjay agarwal.png",
         "caption": "Sanjay Agarwal speaking at the launch", "tags": ["launch"],
         "ocr_text": "", "taken_at": "2024-07-01T00:00:00",
         "taken_at_source": "exif", "taken_at_confidence": 0.9,
         "people": [{"name": "Sanjay Agarwal", "source": "face_filename",
                     "confidence": 0.6}],
         "vectors": {"text": text_vector("sanjay agarwal speaking launch")}},
        {"file_id": "suit1", "sha256": "4" * 64, "type": "image",
         "source_path": "/src/blue-suit.jpg",
         "organised_path": "Organised/2024/blue-suit.jpg",
         "caption": "man in blue suit", "tags": ["blue", "suit"],
         "ocr_text": "", "taken_at": "2024-07-02T00:00:00",
         "taken_at_source": "exif", "taken_at_confidence": 0.9,
         "people": [],
         "vectors": {"text": text_vector("man in blue suit")}},
        {"file_id": "suit2", "sha256": "5" * 64, "type": "image",
         "source_path": "/src/red-suit.jpg",
         "organised_path": "Organised/2024/red-suit.jpg",
         "caption": "suit on a hanger", "tags": ["suit"],
         "ocr_text": "", "taken_at": "2024-07-03T00:00:00",
         "taken_at_source": "exif", "taken_at_confidence": 0.9,
         "people": [],
         "vectors": {"text": text_vector("suit on a hanger")}},
    ]
    for r in recs:
        idx.upsert_file(c, r)
    c.close()
    return db


def test_full_name_query_excludes_other_people(tmp_path):
    """'Yogesh Jain' must not return Yogesh Soni (issue #1)."""
    db = _seed_people(tmp_path)
    hits = searchmod.search(db, "Yogesh Jain")
    assert [h["file_id"] for h in hits] == ["yj"]


def test_person_query_beats_single_keyword_noise(tmp_path):
    """'sanjay sir in blue suit' -> Sanjay first; suit-only rows dropped
    while a named-person match exists (issue #2), ordered semantically."""
    db = _seed_people(tmp_path)
    hits = searchmod.search(db, "sanjay sir in blue suit")
    assert hits, "person row must be returned"
    assert hits[0]["file_id"] == "sanjay"
    if len(hits) > 1:
        assert hits[0]["score"] >= max(h["score"] for h in hits[1:])
    ids = [h["file_id"] for h in hits]
    assert "suit2" not in ids          # lone 'suit' keyword noise excluded


def test_partial_matches_ranked_by_semantic_coverage(tmp_path):
    """No full match -> degrade, but higher term-coverage ranks first."""
    db = _seed_people(tmp_path)
    hits = searchmod.search(db, "blue suit jacket tie")
    assert hits
    # suit1 (blue+suit = 0.5) survives the floor; suit2 (suit only) does not
    assert hits[0]["file_id"] == "suit1"
    assert "suit2" not in [h["file_id"] for h in hits]
    assert all(h["score"] >= hits[-1]["score"] for h in hits)  # sorted desc


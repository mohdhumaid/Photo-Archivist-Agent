"""Flask web UI tests: routes render, search shows thumbnails, images safe."""
from __future__ import annotations

import json
import os

import pytest

pytest.importorskip("flask")


@pytest.fixture()
def web(tmp_path):
    from photo_archivist.web import create_app
    from photo_archivist.core import index as idx
    from PIL import Image

    org = tmp_path / "Organised" / "_Review"
    org.mkdir(parents=True)
    img = org / "sample.jpg"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(img)
    doc = org / "letter.pdf"
    doc.write_bytes(b"%PDF-1.4 fake")

    db = str(tmp_path / "index.db")
    c = idx.connect(db)

    img_rec = {
        "file_id": "aabbccddeeff", "sha256": "f" * 8, "type": "image",
        "mime": "image/jpeg", "organised_path": str(img),
        "source_path": str(tmp_path / "src.jpg"),
        "caption": "Branch launch photo", "tags": ["launch"],
        "taken_at": "2025-07-14T10:00:00", "taken_at_confidence": 0.9,
        "place": {"value": "Indiranagar", "source": "path", "confidence": 0.4},
        "people": [{"name": "Unknown Person 1", "source": "face_unmatched",
                    "confidence": 0.2, "box": [1, 2, 3, 4]}],
        "vectors": {"text": None, "image": None},
    }
    doc_rec = {
        "file_id": "112233445566", "sha256": "e" * 8, "type": "document",
        "mime": "application/pdf", "organised_path": str(doc),
        "source_path": str(doc), "caption": "Sanction letter 2023",
        "tags": ["sanction"], "people": [],
        "vectors": {"text": None, "image": None},
    }
    idx.upsert_file(c, img_rec)
    idx.upsert_file(c, doc_rec)
    c.execute("INSERT INTO folders(name, tags, people, place, date_from, date_to, pattern) "
              "VALUES(?,?,?,?,?,?,?)",
              ("2025-07 Branch Launches", json.dumps(["launch"]), "[]",
               "Indiranagar", "2025-07-01", "2025-07-31", "yyyy-mm Event"))
    c.commit()
    c.close()

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "index_db: %s\norganised_dir: %s\npeople: {}\nroles: {}\n"
        "faces:\n  library: %s\nvision_llm:\n  enabled: false\n"
        % (db, str(tmp_path / "Organised"), str(tmp_path / "faces_library.json")),
        encoding="utf-8")
    app = create_app(str(cfg))
    app.config["TESTING"] = True
    return app.test_client(), img_rec, doc_rec


def test_all_pages_render_200(web):
    client, _img, _doc = web
    for url in ("/", "/browse", "/browse?type=image", "/search",
                "/search?q=sanction", "/people", "/folders", "/review",
                "/file/aabbccddeeff"):
        resp = client.get(url)
        assert resp.status_code == 200, (url, resp.status_code)


def test_search_renders_image_thumbnail(web):
    client, img_rec, _doc = web
    resp = client.get("/search?q=launch")
    html = resp.get_data(as_text=True)
    assert resp.status_code == 200
    assert f"/thumb/{img_rec['file_id']}" in html      # image rendered inline
    assert "Branch launch photo" in html
    assert "reason" in html or "from path" in html     # provenance shown


def test_thumb_serves_organised_image(web):
    client, img_rec, _doc = web
    resp = client.get(f"/thumb/{img_rec['file_id']}")
    assert resp.status_code == 200
    assert resp.mimetype == "image/jpeg"
    assert len(resp.data) > 0


def test_document_thumb_gets_placeholder_not_bytes(web):
    client, _img, doc_rec = web
    resp = client.get(f"/thumb/{doc_rec['file_id']}")
    assert resp.status_code == 200
    assert "svg" in resp.mimetype


def test_missing_file_thumb_is_placeholder_not_500(web, tmp_path):
    client, img_rec, _doc = web
    # delete the image behind the DB's back -> placeholder, never a crash
    os.remove(img_rec["organised_path"])
    resp = client.get(f"/thumb/{img_rec['file_id']}")
    assert resp.status_code == 200
    assert "svg" in resp.mimetype


def test_bad_file_id_and_traversal_rejected(web):
    client, _img, _doc = web
    for bad in ("nothex", "ZZZZZZ", "../../etc/passwd", "..%2f..%2fconfig.yaml",
                "aabbccddeeff'--"):
        resp = client.get(f"/thumb/{bad}")
        assert resp.status_code in (404, 400), (bad, resp.status_code)


def test_unknown_file_id_404(web):
    client, _img, _doc = web
    assert client.get("/thumb/abcdef123456").status_code == 404
    assert client.get("/file/abcdef123456").status_code == 404


def test_download_returns_attachment_with_bytes(web):
    client, img_rec, doc_rec = web
    resp = client.get(f"/file/{img_rec['file_id']}/download")
    assert resp.status_code == 200
    assert "attachment" in resp.headers.get("Content-Disposition", "")
    assert "sample.jpg" in resp.headers.get("Content-Disposition", "")
    assert len(resp.data) > 0
    # documents download too
    resp2 = client.get(f"/file/{doc_rec['file_id']}/download")
    assert resp2.status_code == 200
    assert "attachment" in resp2.headers.get("Content-Disposition", "")


def test_download_unknown_or_bad_id_404(web):
    client, _img, _doc = web
    assert client.get("/file/abcdef123456/download").status_code == 404
    assert client.get("/file/nothex/download").status_code == 404


def test_detail_page_shows_download_button(web):
    client, img_rec, _doc = web
    html = client.get(f"/file/{img_rec['file_id']}").get_data(as_text=True)
    assert f"/file/{img_rec['file_id']}/download" in html


def test_cards_show_download_links(web):
    client, _img, _doc = web
    html = client.get("/browse").get_data(as_text=True)
    assert "/download" in html


def test_review_lists_unknown_person(web):
    client, _img, _doc = web
    html = client.get("/review").get_data(as_text=True)
    assert "Unknown Person 1" in html

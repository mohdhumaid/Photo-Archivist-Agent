"""Live-upgrade tests: folder learning, gazetteer geocode, embed backends,
face library loading, and graceful fallbacks when optional AI is missing."""
import json
import os

from photo_archivist.core import embed as embedmod
from photo_archivist.core import index as idx
from photo_archivist.core import people as peoplemod
from photo_archivist.core import place as placemod
from photo_archivist.core import decide as decmod


# --- folder learning ---------------------------------------------------------

def _rec(fid, vec, tags, name_inbox="Inbox"):
    return {"file_id": fid, "sha256": fid * 8, "source_path": f"/src/{fid}.docx",
            "organised_path": f"Organised/{name_inbox}/{fid}.docx", "type": "document",
            "caption": "x", "tags": tags, "ocr_text": "",
            "place": {"value": "Indiranagar", "source": "path"},
            "event": {"value": None}, "taken_at": "2025-07-14T10:00:00",
            "taken_at_source": "exif", "taken_at_confidence": 0.99,
            "people": [], "vectors": {"text": vec, "image": None}}


def test_folder_profile_roundtrip_and_promote(tmp_path):
    db = str(tmp_path / "i.db")
    vec = embedmod.text_vector("loan sanction letter 2023 branch", backend="hash")
    recs = [_rec("f1", vec, ["branch", "letter"]), _rec("f2", vec, ["branch", "letter"])]
    prof = idx.profile_from_records("Inbox", recs)
    assert prof["name"] == "Inbox" and prof["centroid"] and prof["tags"]
    c = idx.connect(db)
    idx.upsert_folder(c, prof)
    c.close()
    loaded = idx.load_folders(db)
    assert len(loaded) == 1 and loaded[0]["name"] == "Inbox"
    assert len(loaded[0]["centroid"]) == len(vec)
    # a near-identical new file must now be PROMOTED into the learned folder
    d = decmod.decide(vec, {"branch", "letter"}, set(), loaded,
                      promote=0.80, review_low=0.60)
    assert d["action"] == "file" and d["folder"] == "Inbox"


def test_load_folders_missing_db(tmp_path):
    assert idx.load_folders(str(tmp_path / "nope.db")) == []


# --- reverse geocoding -------------------------------------------------------

def test_gazetteer_load_and_lookup(tmp_path):
    tsv = tmp_path / "cities.tsv"
    # GeoNames layout: id name asciiname alternatenames lat lon fclass fcode country cc2 admin1
    tsv.write_text("1273865\tBengaluru\tBangalore\t Bengaluru\t12.97194\t77.59369\tP\tPPL\tIN\t\tKA\n",
                   encoding="utf-8")
    db = str(tmp_path / "gazetteer.db")
    assert placemod.load_gazetteer(str(tsv), db) == 1
    out = placemod.reverse_geocode(12.97, 77.59, db_path=db)
    assert out["source"] == placemod.OFFLINE_SOURCE and out["confidence"] == 0.9
    assert "Bengaluru" in out["value"]


def test_geocode_stub_without_db(tmp_path):
    out = placemod.reverse_geocode(12.97, 77.59, db_path=str(tmp_path / "missing.db"))
    assert out["source"] == placemod.STUB_SOURCE and out["confidence"] == 0.5


# --- embeddings --------------------------------------------------------------

def test_hash_backend_deterministic():
    a = embedmod.text_vector("loan sanction letter", backend="hash")
    b = embedmod.text_vector("loan sanction letter", backend="hash")
    assert a == b and len(a) == embedmod.HASH_DIM


def test_auto_backend_returns_vector():
    v = embedmod.text_vector("branch launch indiranagar", backend="auto")
    assert isinstance(v, list) and len(v) > 0


# --- face library ------------------------------------------------------------

def test_face_library_missing_and_invalid(tmp_path):
    assert peoplemod.load_face_library(str(tmp_path / "none.json")) == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert peoplemod.load_face_library(str(bad)) == {}


def test_match_local_below_threshold():
    assert peoplemod.match_local([1.0, 0.0], {"X": [0.0, 1.0]}, 0.6) is None


# --- vision fallback ---------------------------------------------------------

def test_local_backend_always_resolves_to_mock():
    from photo_archivist.core import vision as v
    # org policy: no vision_local module ships, so "local" can never load a model
    assert isinstance(v.get_backend("local"), v.MockVision)
    assert isinstance(v.get_backend("local", {}), v.MockVision)


# --- e2e: scan learns folders, second scan promotes --------------------------

def test_scan_apply_then_promote(tmp_path):
    from typer.testing import CliRunner
    from photo_archivist.cli import app
    from docx import Document

    src = tmp_path / "src"
    src.mkdir()
    for i, words in enumerate(["loan sanction letter 2023", "letter loan sanction 2023"]):
        d = Document()
        d.add_paragraph(words)
        d.save(str(src / f"doc{i}.docx"))
    cfgp = tmp_path / "config.yaml"
    cfgp.write_text(
        f"index_db: {tmp_path / 'i.db'}\n"
        f"organised_dir: {tmp_path / 'Org'}\n"
        f"undo_log: {tmp_path / 'undo.log'}\n"
        f"embeddings:\n  backend: hash\n")
    runner = CliRunner()
    r1 = runner.invoke(app, ["scan", str(src), "--no-dry-run", "--yes",
                             "--config", str(cfgp)])
    assert r1.exit_code == 0, r1.output
    # no learned profiles on first run: metadata-derived folder (spec §5),
    # NOT a hardcoded Inbox (only _Review bypasses new-folder creation)
    profiles = idx.load_folders(str(tmp_path / "i.db"))
    assert profiles and profiles[0]["name"] not in ("Inbox", "_Review")
    assert "WROTE 2 copies" in r1.output
    # second scan: identical tags/vectors -> promote (action=file into Inbox)
    r2 = runner.invoke(app, ["scan", str(src), "--no-dry-run", "--yes",
                             "--config", str(cfgp)])
    assert r2.exit_code == 0, r2.output
    assert "Loaded 1 folder profile(s)" in r2.output

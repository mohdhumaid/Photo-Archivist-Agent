# Photo Archivist Agent

Bank-grade photo/document organiser. Read-only source, copies only, dry-run first,
every write logged + undoable. Full operating rules: see the spec PDF
(`OPERATING_CONTEXT.md` once added).

## Quickstart

```bash
pip install Pillow ImageHash numpy pypdf pikepdf python-docx openpyxl olefile typer pyyaml pytesseract pytest
brew install exiftool tesseract poppler   # macOS; apt on Linux

# 1. Dry run (writes nothing)
python -m photo_archivist.cli scan /path/to/source

# 2. Apply after approving the plan
python -m photo_archivist.cli scan /path/to/source --no-dry-run --yes

# 3. Search (every hit carries reason + provenance)
python -m photo_archivist.cli search "Indiranagar branch launch"
python -m photo_archivist.cli search "loan sanction letters from 2023"

# 4. Undo last batch
python -m photo_archivist.cli undo
```

Optional local-AI extras (`pip install -e .[ai-local]`): insightface,
sentence-transformers, open-clip-torch, torch. The default `mock` vision
backend makes zero claims; swap in `photo_archivist/core/vision_local.py`
following the `VisionBackend` interface. Faces never leave the machine.

## Layout

```
config.yaml             # all tunable DEFAULTs (thresholds, excludes, hardlink, models)
people_library.yaml     # confirmed names + org role map (MD -> Anita Rao ...)
photo_archivist/cli.py  # scan / search / undo
photo_archivist/core/
  detect.py fingerprint.py pipeline.py
  metadata/{exif,docs,video,fs,reconcile,place}.py
  text.py vision.py people.py place.py embed.py
  index.py decide.py organise.py search.py report.py pii.py
tests/  sample_data/  data/
```

Writes go to `Organised/` + `index.db` + sidecar `.tags.json` + `undo.log` only.
# Photo-Archivist-Agent

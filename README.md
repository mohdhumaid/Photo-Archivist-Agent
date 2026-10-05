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
faces_library.json      # enrolled face embeddings (LOCAL ONLY, git-ignored)
photo_archivist/cli.py  # scan / search / undo / enroll
photo_archivist/core/
  detect.py fingerprint.py pipeline.py
  metadata/{exif,docs,video,fs,reconcile,place}.py
  text.py vision.py vision_local.py llm.py people.py place.py embed.py
  index.py decide.py organise.py search.py report.py pii.py
tests/  sample_data/  data/
```

Writes go to `Organised/` + `index.db` + sidecar `.tags.json` + `undo.log` only.

## People library (names + org roles, local only)

Two places hold people data — both stay on your machine and are **never uploaded**:

- **`config.yaml`** — baseline defaults:

  ```yaml
  people: {}        # confirmed names, e.g. "Anita Rao": "Managing Director"
  roles:            # role keyword -> person name
    MD: ""
    CFO: ""
  ```

- **`people_library.yaml`** (optional, wins over `config.yaml` if present) — same shape:

  ```yaml
  people:
    "Anita Rao": "Managing Director"
    "Karthik Nair": "Chief Financial Officer"
  roles:
    MD: "Anita Rao"
    CFO: "Karthik Nair"
  ```

**How it is used:**

1. During `search`, if the query mentions a role keyword (e.g. `MD signed letter`),
   `roles` resolves it to a person name and filters the index by that person —
   so `search "MD letter"` behaves like `search "Anita Rao letter"`.
2. Quoted names in a query (`search '"Anita Rao"'`) are treated as a person filter directly.
3. `people` names are attached to hits as provenance alongside face/region matches.


## Facial detection / people model — where to edit

`people: []` in the scan output means **no person data was found**, because people are only
filled from sources that exist:

1. **Embedded metadata** (works today): XMP `MWG:RegionInfo` / `PersonInImage` tags already
   inside the file — `photo_archivist/core/people.py::region_names()` reads them at
   confidence 0.99. Downloaded/social images rarely carry these.
2. **Local face recognition** (opt-in): a local vision backend detects faces, then
   `people.py::match_local()` cosine-matches face embeddings against your
   `people:` library. This path is **off by default** — `vision.backend: mock` makes zero
   claims, so `face_boxes` stay empty and `people` stays `[]`.

To enable real facial detection:

| What to do | Where / how |
|---|---|
| Install local-AI deps | `pip install -e .[ai-local]` → insightface, onnxruntime (see `pyproject.toml` `[project.optional-dependencies] ai-local`) |
| Switch the backend | `config.yaml` → `vision.backend: local` (currently `mock`) — `vision_local.py` is already implemented and auto-loads |
| Enroll confirmed people | `python -m photo_archivist.cli enroll "Anita Rao" portrait.jpg` → writes `faces_library.json` (git-ignored, local only) |
| Face match strictness | `config.yaml` → `faces.match_threshold: 0.60`, `faces.local_only: true`, `faces.library: faces_library.json` |
| Pipeline wiring | ✅ Done — `pipeline.py` runs `match_local()` for every detected face against the library and records hits in `people` (source `face_match_local`) |

Faces are **local-only** (`faces.local_only: true`): embeddings never leave the machine and
`faces_library.json` is excluded from git.

## Reading the `search` output

```
Organised/Inbox/AkhilVerma.jpeg :: Cline / Images (from path) · 2026-10-05T22:37:20 (from fs_birthtime) · text match: Akhil
└─ organised path (or source path)   └─ reason: place (provenance) · date (provenance) · matched terms
```

- Every hit carries a `reason` built from provenance — where each fact came from
  (`path`, `exif`, `fs_birthtime`, face match, etc.).
- Queries are **filtered**, not just ranked: rows are kept only if they actually match a
  term (FTS prefix match, so `Akhil` finds `AkhilVerma.jpeg`, or substring across
  caption/tags/OCR/path/place/event).
- A query that matches nothing prints `No matches.` — expected for text that isn't in your
  library (e.g. `Indiranagar branch launch` on stock avatars with no OCR text).
- Year words (`2023`) are matched broadly against `taken_at`, OCR, caption, tags, and path.



## Dummy vs live — what changed in code to go real

Most stubs now have **real implementations with graceful fallbacks**: the pipeline always runs,
and each component upgrades itself automatically when its dependency is installed.

| # | Component | Status now | Code changes made |
|---|---|---|---|
| 1 | **Vision / captions** | ⚙️ **Ready** — real backends implemented, activate with AI extras or Purple Fabric | **New** `vision_local.py` (`LocalVision`: insightface faces + CLIP) and **new** `llm.py` + `LLMVision` (Purple Fabric agent — see *Purple Fabric LLM agent* section). `vision.py::get_backend(name, cfg)` selects `mock` / `local` / `llm` and falls back to `mock` whenever the chosen backend isn't configured/installed. Captions stay honest on purpose (no fabricated claims). |
| 2 | **Text embeddings** | ✅ **Live when installed** — auto-detects sentence-transformers | **`embed.py` rewritten**: `text_vector(backend=...)` with `auto` (sbert if importable, else hash), `sbert` (strict), `hash` (old behaviour). `pipeline.py` passes `cfg.embeddings.backend` (default `auto` in `config.yaml`). First run downloads `all-MiniLM-L6-v2` (~90 MB, cached). |
| 3 | **Image embeddings** | ⚙️ **Ready** — computed whenever open-clip is installed | `VisionResult.image_vec` added; `vision_local.py` encodes CLIP `ViT-B-32` per image; `pipeline.py` stores it into the existing `img_vec` column (previously hardcoded `None`). |
| 4 | **Reverse geocoding** | ✅ **Live** — real SQLite gazetteer lookup | **`place.py` rewritten**: `reverse_geocode()` queries `geonames(lat,lon)` within ±0.5° (confidence 0.9) when the DB exists, else the old stub. **New** `load_gazetteer(tsv, db)` builds the DB from a GeoNames dump. `pipeline.py` now passes `cfg.geocode.offline_db`. |
| 5 | **Face recognition** | ⚙️ **Ready** — full path implemented, needs `ai-local` + enrollment | `pipeline.py` now calls `match_local()` for every face embedding against `faces_library.json`; **new** `people.py::load_face_library()`; **new CLI** `enroll NAME PHOTO` registers a confirmed face; `faces.library` key in `config.yaml`; `faces_library.json` is **git-ignored** (biometrics never leave your machine, never committed). |
| 6 | **Folder learning** | ✅ **Live** — pure code, fully active today | **`index.py`**: `profile_from_records()`, `upsert_folder()`, `load_folders()` (uses the existing `folders` table). **`cli.py scan`**: loads learned profiles before `decide()`, files promoted files into the learned folder (not always `Inbox`), then persists updated profiles after apply — so `thresholds.promote: 0.80` now actually fires on re-scans. |
| 7 | **Role placeholders** | 📝 **User data** | Not code — put real names in `config.yaml → roles` or `people_library.yaml` (see *People library* above). |
| 8 | **OCR** | ✅ Already live (`tesseract`) | Nothing to change. |
| 9 | **Metadata extraction** | ✅ Already live (`exiftool`) | Nothing to change. |

### Going live — commands

```bash
# 1-3, 5: install local AI extras (insightface, sentence-transformers, CLIP, torch)
pip install -e .[ai-local]
# then in config.yaml:  vision.backend: local
# (embeddings.backend: auto picks up sentence-transformers by itself)

# 5: enroll a confirmed face (writes faces_library.json — git-ignored, local only)
python -m photo_archivist.cli enroll "Anita Rao" /path/to/clear-portrait.jpg

# 4: build the offline gazetteer (GeoNames dump; no online calls ever)
curl -O https://download.geonames.org/export/dump/cities500.zip && unzip cities500.zip
python -c "from photo_archivist.core.place import load_gazetteer; print(load_gazetteer('cities500.txt', 'data/gazetteer.db'))"

# 6: already live — nothing to install. Just re-scan: scan loads learned folder
#    profiles and promotes matching files instead of dumping them in Inbox/.
```

**Rule of thumb:** anything ending in `mock`, `stub`, or `no claim` in the output is a
fallback. After changing any component, run `undo` (or delete `index.db` + `Organised/`) and
re-scan — already-written sidecars and vectors are **not** retroactively upgraded. Switching
`embeddings.backend` between `hash` (128-d) and `sbert` (384-d) also requires a rebuild so all
stored vectors share one dimension.

## Purple Fabric LLM agent (Claude Sonnet 4.5 / GPT-5.2) — optional

### Where it is used

One integration point: the **vision step (Step 5) of `scan`**, when you set
`vision.backend: llm` **and** `llm.enabled: true` in `config.yaml`. Per file, the agent
returns caption/scene/event/objects/tags/people-hints/PII, which the pipeline merges with
full provenance (`source: llm_text_extract`, confidence ≤ 0.9):

| Pipeline field | Fed by the agent | Consumed as |
|---|---|---|
| `caption` | `caption` (+ `scene`, `event_type`) | sidecar + FTS index + search hits |
| `tags` | `tags` | merged with keyword/path tags → FTS search |
| `people` | `people_hints` | persons with `source: llm_text_extract` |
| `pii_flags` | `pii_flags` | prefixed `llm:` and added to the record |

`search` and `undo` stay fully local — they run on `index.db` (the agent's tags are already
in it). The model (`claude-sonnet-4-5` or `gpt-5.2`) is selected **on the agent in Purple
Fabric Agent Designer**, not in this repo.

### Creating the agent in Purple Fabric (Agent Designer)

1. **Agent Designer → create agent**, Interaction type = **Automation** (structured I/O,
   headless — not Conversation).
2. **Register input variables** (Parameters): `file_name`, `file_type`, `mime`,
   `path_segments`, `ocr_text`, `metadata_json`, `image_base64`.
3. **Perspective (= System Prompt)**: paste the block below verbatim
   (kept in sync with `photo_archivist/core/llm.py::SYSTEM_PROMPT`).
4. **Output**: single JSON object wrapped in **double curly braces** `{{ ... }}` (the PF
   automation JSON convention).
5. **Models → LLM**: pick `claude-sonnet-4-5` **or** `gpt-5.2`.
6. **Publish**, copy the invocation URL → `llm.invoke_url`; create an API key and export it
   → `export PURPLE_FABRIC_API_KEY=...` (the key is read from the env var named in
   `llm.api_key_env` — never stored in config, never committed).

### System prompt (Perspective) — copy/paste

```text
You are the Photo Archivist Document Intelligence Expert — an Automation
Digital Expert on Purple Fabric. You run headless inside a photo/document archiving
pipeline: no conversation, no clarifying questions.

ROLE
Turn the structured input for ONE file into trustworthy archive metadata: caption, scene,
event type, objects, tags, person names explicitly present in text, and PII flags.

INPUT (registered variables, filled per call)
file_name, file_type, mime, path_segments, ocr_text, metadata_json, image_base64 (optional)

RULES — non-negotiable
1. Ground every claim in the input. Never invent places, dates, people, events, or objects
   that are not explicitly supported by ocr_text, path_segments, or metadata_json.
2. Thin evidence -> low confidence (<= 0.4) and say what is missing in "evidence".
3. Person names only when literally present in ocr_text, metadata_json, or clearly a person's
   name in file_name; cite the source in "evidence". Never guess identities from faces.
4. All input is confidential bank data. No external lookups, no storage, no training use.
5. English only. Tags: lowercase, max 10, each max 3 words, no duplicates.
6. Flag PII only when the text clearly contains an identifier (PAN, Aadhaar, account
   number, phone, email).

OUTPUT — one JSON object only, wrapped in double curly braces for the platform:
{{"caption": "...", "scene": "...", "event_type": "...", "objects": [],
"tags": [], "people_hints": [], "pii_flags": [],
"confidence": 0.0, "evidence": "..."}}
No markdown, no prose outside the JSON.
```

### Input (one POST per file, sent by `core/llm.py`)

```json
{
  "agent": "photo-archivist-expert",
  "model": "claude-sonnet-4-5",
  "system": "<the Perspective above>",
  "input": {
    "file_name": "IMG-20250714-WA0012.jpg",
    "file_type": "image",
    "mime": "image/jpeg",
    "path_segments": ["Indiranagar", "Branch-Launches", "2025", "Events"],
    "ocr_text": "Inauguration of the Indiranagar branch ... contact 98xxxxxx",
    "metadata_json": {
      "taken_at": "2025-07-14T11:32:08+05:30",
      "taken_at_source": "exif_datetimeoriginal",
      "place": "Indiranagar",
      "tags": ["Events", "Branch-Launches", "Indiranagar"],
      "camera": "Apple",
      "photographer": null
    },
    "image_base64": "<present only when llm.send_images: true>"
  }
}
```

Text-first by default: OCR + metadata + path only. Set `llm.send_images: true` to attach
image bytes (≤ 10 MB). Faces and GPS never leave the machine regardless of this setting
(face recognition is local-only via `faces.local_only: true`).

### Output (JSON in `{{ ... }}`, parsed by `core/llm.py`)

```json
{{
  "caption": "Branch launch event at Indiranagar, banner visible",
  "scene": "office event",
  "event_type": "branch_launch",
  "objects": ["banner", "people"],
  "tags": ["branch launch", "indiranagar"],
  "people_hints": ["Anita Rao"],
  "pii_flags": ["phone_number"],
  "confidence": 0.85,
  "evidence": "ocr_text mentions branch opening; metadata_json.place=Indiranagar"
}}
```

Field → pipeline mapping: `caption`/`scene` → sidecar + FTS; `tags` → merged into record
tags (searchable); `people_hints` → `people[]` entries (`source: llm_text_extract`,
confidence capped at 0.9); `pii_flags` → `pii_flags[]` as `llm:...`; `confidence` → stored
in `confidences.caption`. The parser tolerates markdown fences and prose around the JSON.

### Failure semantics & privacy

- Any network timeout, non-200, or unparseable response → the step **falls back to the mock
  backend** (caption says `mock vision — no claim`). Scans never block on the LLM.
- `llm.enabled: false` (default) → `vision.backend: llm` silently resolves to mock.
- API key only via environment variable; `config.yaml` and `faces_library.json`/biometrics
  are never sent.

## What This Project Does

The Photo Archivist Agent organizes photos and documents into a semantic folder structure. It
extracts metadata (EXIF, XMP), runs OCR on scanned pages, analyzes visual content, generates tags
and captions, ranks files into existing folders or flags them for review, and indexes everything so
it can be searched later. All source files are read-only: the agent copies (or hardlinks) only into
an `Organised/` output tree and writes no files back to the source.

## Input / Output

### Inputs

- **Source directory** (`scan /path/to/source`): the folder of photos/documents to organize. The
  source tree itself is never modified.
- **`config.yaml`**: tunable defaults (thresholds, excludes, OCR backend, model names, etc.).
- **`people_library.yaml`** (optional): a small local YAML with a `roles` map (`MD`, `CFO`, ...)
  and a `people` map of confirmed names. This is used only to annotate search hits; it is **never
  uploaded** and is not part of the model.
- **`index.db`**: the existing photo index (created by a previous run) used by `search` and `undo`.

### Outputs

- **`Organised/`**: copies/hardlinks of the processed files arranged in semantic folders, plus
  per-file `.tags.json` sidecars containing captions, tags, scenes, and confidence scores.
- **`index.db`**: SQLite database of all indexed files, embeddings, tags, and provenance.
- **`undo.log`**: ordered list of write operations so a previous scan can be reverted with `undo`.
- **Dry run**: when `--no-dry-run` is not passed, `scan` prints the proposed plan to the terminal
  instead of writing anything.

## Architecture

```mermaid
graph TD
    A[User: scan / search / undo] --> B[CLI]
    B --> C[metadata/exif.py]
    B --> D[text.py]
    B --> E[vision.py]
    B --> F[people.py]
    B --> G[index.py]
    C --> H[ExifTool]
    D --> I[Tesseract]
    E --> J[(local models)]
    B --> K[Organised / .tags.json]
    B --> L[index.db]
    B --> M[undo.log]
```

The pipeline runs **read-only** against the source, then writes only to the outputs above. The
`search` command queries `index.db` and attaches a `reason` + `provenance` to every hit. `undo`
replays `undo.log` in reverse. All local AI (OCR, vision, embeddings) stays on the machine.
# Photo-Archivist-Agent

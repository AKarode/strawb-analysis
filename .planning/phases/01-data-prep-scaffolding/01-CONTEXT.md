# Phase 1: Data Prep & Scaffolding — Context

**Gathered:** 2026-04-23
**Status:** Ready for planning
**Source:** User-directed scoping (auto mode, skipped discuss-phase)

<domain>
## Phase Boundary

Phase 1 produces the authoritative data layer and repo skeleton that Phases 2–5 consume. Everything downstream (detection training, disease training, CPU pipeline, Hailo port) reads from this phase's outputs.

**In scope:**
- Project skeleton directories (`src/`, `models/`, `reports/`) and placement of loose artifacts currently in repo root (the pretrained `yolo26n.pt`, `yolo26n-cls.pt`, and their `*_ncnn_model/` exports).
- `data/MANIFEST.json` — per-image provenance, license, hash, annotation type, class assignment, split.
- Deterministic dedup audit across datasets (OSF ↔ Kaggle ↔ afzaal-bbox-v4 overlaps) producing `reports/dedup-report.md`.
- `data/disease_crops/{train,val,test}/<class>/` — per-polygon padded fruit-disease crops ready for YOLO26n-cls training.
- `LICENSE.md` per dataset folder, especially calling out StrawDI's non-commercial-academic constraint.

**Explicitly out of scope (user directive):**
- Re-organizing, re-downloading, or restructuring existing `data/` subdirectories. What's on disk stays on disk.
- Writing `data.yaml` files from scratch — verify existing ones instead.
- Training, inference, or evaluation code. All downstream.

</domain>

<decisions>
## Implementation Decisions

### Locked by user (scope narrowing)
- **Skeleton only what's missing** — create `src/`, `models/`, `reports/`; do NOT touch existing `data/` subtree organization.
- **Move pretrained weights** — `yolo26n.pt`, `yolo26n-cls.pt`, `yolo26n_ncnn_model/`, `yolo26n-cls_ncnn_model/` migrate from repo root into `models/detect/` and `models/disease/` (or equivalent sub-paths). Update any script references.
- **MANIFEST.json is authoritative** — every downstream phase reads image metadata from this file, not by walking the filesystem. Per-image fields: source dataset, original filename, license, annotation type (bbox/polygon/mask), project-schema class, split, SHA256 hash.
- **Dedup prefers canonical copy** — when byte-identical files appear across datasets (e.g., OSF ↔ Kaggle confirmed duplicates), MANIFEST points at one canonical path; the rest are recorded as aliases. Does NOT delete files on disk.
- **disease_crops from polygons, not full frames** — each Kaggle polygon gets a padded bounding box (padding TBD by planner; sensible default ~10–15%) cropped out. Leaf-disease classes either excluded or partitioned into a separate `leaf_*` subtree clearly marked as non-training.
- **Determinism** — re-running any conversion script on the same inputs produces byte-identical outputs. Fixed seed wherever sampling occurs.

### Locked by project (CLAUDE.md / PROJECT.md)
- Python 3.11+ project; use existing `.venv/`.
- All data + model paths are gitignored; this phase ships scripts and the MANIFEST, not data.
- Hailo hardware is Hailo-10H (NOT Hailo-8 as PROJECT.md currently says) — memory flag; not a Phase 1 concern but scripts should be Hailo-agnostic.

### Claude's Discretion
- MANIFEST.json exact schema shape (flat array vs. keyed-by-hash) — pick what's easiest to query.
- Which scripting convention: single `prepare_data.py` monolith vs. one script per dataset (e.g., `build_manifest.py`, `dedup_audit.py`, `build_disease_crops.py`). Favor the latter for reproducibility + testability.
- Dedup hash: SHA256 over MD5 (SHA256 is fine; MD5 is fine for this). Pick SHA256 for future-proofing.
- Whether to zap leaf-disease classes entirely or keep them in a `data/disease_crops/_leaves_excluded/` holdout. Default: holdout.
- Synthesized `healthy` class construction — Phase 3 concern, NOT this phase. Phase 1 just ensures the infrastructure is there (disease_crops layout supports adding `healthy/` later).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project spec
- `.planning/PROJECT.md` — project goals, constraints, hardware assumption
- `.planning/REQUIREMENTS.md` — DATA-01..07 (the REQ-IDs this phase owns)
- `.planning/ROADMAP.md` — full roadmap including Phase 1 success criteria
- `CLAUDE.md` — repo layout conventions, environment setup
- `agri-project-goals.docx` — original client spec (reference only)

### In-session research (use as research artifacts — do NOT re-research)
- `.planning/research/SUMMARY.md` — verified technology landscape
- `.planning/research/2026-04-21-dataset-inventory-and-splits.md` — firsthand disk inspection, class counts, resolution samples, collision detection method
- `.planning/research/2026-04-21-verification-and-dataset-inspection.md` — earlier verification pass
- `.planning/research/2026-04-21-evaluator-csv-schema.md` — downstream CSV schema (Phase 4 concern but relevant for MANIFEST schema parity)
- `.planning/research/2026-04-21-yolo26-vs-yolo12-pi5.md` — model choice justification

### Live state
- Current `data/` subtree on disk (see `ls data/`):
  - `data/zenodo/strawberries/` (extracted; 813 images, YOLO bbox, ripe/unripe/peduncle)
  - `data/disease/` (Kaggle Afzaal, polygon, 7 disease classes)
  - `data/osf-ej5qv/` (OSF re-host of Kaggle disease — known partial duplicate)
  - `data/roboflow/{afzaal-bbox-v4, matt-lucky-ripeness, research-proj-disease}/`
  - `data/strawdi/extracted/` (StrawDI_Db1 masks)
- Pretrained weights currently loose in repo root: `yolo26n.pt`, `yolo26n-cls.pt`, `yolo26n_ncnn_model/`, `yolo26n-cls_ncnn_model/`
- Existing scripts in `scripts/`: `smoke_test_yolo26.py`, `pi/` benchmark tooling

</canonical_refs>

<specifics>
## Specific Ideas

- Known duplicate: `anthracnose_fruit_rot1.jpg` was MD5-confirmed across OSF and Kaggle disease folders during prior research. The dedup script should reproduce and extend this finding across all pairs.
- MANIFEST schema should support Phase 4's evaluator consuming it for ground-truth lookup — keep it queryable by image hash and by (source, filename).
- `data/strawdi/` LICENSE.md must explicitly call out "non-commercial academic use only" and mark it as detection-only (not redistributable in a commercial product).

</specifics>

<deferred>
## Deferred Ideas

- **Synthesized `healthy` class generation** — Phase 3 responsibility; the classifier trainer samples non-diseased Zenodo crops to build it. Phase 1 only ensures `data/disease_crops/` layout can accept a later-added `healthy/` class folder.
- **Train/val/test split ratios for disease_crops** — delegate to planner, but default to preserving Kaggle's source splits rather than reshuffling. Confirm during planning if deviation is warranted.
- **Phase 1.5 Hailo harness (open scope question #4)** — not decided; does not block Phase 1.

</deferred>

---

*Phase: 01-data-prep-scaffolding*
*Context gathered: 2026-04-23 via auto-scoped path (user directed narrowing)*

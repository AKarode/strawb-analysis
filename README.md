# Strawberry Vision Pi

Offline batch image-analysis pipeline for strawberry field photos, deployed on a Raspberry Pi 5 with the AI HAT+ 2 (Hailo-10H, 40 TOPS).

Given a folder of 500–1000 RGB field images, it produces a single CSV with per-image counts (ripe / unripe / peduncle), per-fruit disease predictions, and stage timings — plus an offline evaluator that compares the CSV against ground-truth annotations to report detection mAP and per-class disease accuracy.

**Not** real-time / streaming. **Not** multispectral (the legacy `multi-spectral` folder name is a misnomer — input is plain RGB). **Not** an LLM project.

## How it works

Two-stage detect-then-classify pipeline:

```
images/  →  YOLO26n (detect)  →  crop each fruit  →  YOLO26n-cls (disease)  →  per-image CSV
                                                                                  │
                                                                                  ▼
                                                                          evaluator → mAP, per-class accuracy
```

Detector and classifier are trained separately because the source datasets cannot be merged into one segmentation model — they have no image overlap and incompatible label schemas.

Two backends share one entry point (`run_inference.py`):

| Backend | Runtime | Use case |
|---|---|---|
| `cpu` | NCNN FP32 @ 640×640 | Reference path, no accelerator needed |
| `hailo` | Hailo HEF on AI HAT+ 2 | Production target, ~70× under the 30-min budget for 1000 images |

INT8 on NCNN is **not** viable on Pi 5 as of April 2026 — CPU path stays FP32.

## Hardware

- Raspberry Pi 5 Model B Rev 1.1, 16 GB RAM
- AI HAT+ 2 with Hailo-10H (40 TOPS)
- Access via Raspberry Pi Connect (no SSH)

On-device benchmark, 2026-04-23, against a 48-image folder:

| Model | 48-img wall | ms/img |
|---|---|---|
| yolov8n | 2.42 s | ~50 |
| yolov8s | 2.49 s | ~52 |
| yolov8m | 2.48 s | ~52 |
| yolov6n | 2.45 s | ~51 |
| yolov11n | 2.42 s | ~51 |

Decomposes to ~1.3 s startup + ~24 ms / image. **~25 s for 1000 images** end-to-end. Bottleneck is Python + preprocessing, not the Hailo NPU — so model selection is accuracy-driven, not throughput-driven.

## Repository layout

```
.planning/                     # GSD planning workflow — source of truth for decisions
  PROJECT.md                   # context, core value, constraints, key decisions
  REQUIREMENTS.md              # 29 v1 requirements across DATA / DETECT / DISEASE / PIPE / HAILO / EVAL
  ROADMAP.md                   # 5-phase plan
  STATE.md                     # current phase + progress
  research/                    # verified technology landscape (datasets, YOLO26, Hailo)
  phases/01-data-prep-scaffolding/   # current phase

src/                           # library code
  manifest/                    # per-image provenance: schema, hashing, dataset scanners

scripts/                       # CLIs (training, conversion, manifest builders)
  build_manifest.py            # walks data/, emits data/MANIFEST.json
  dedup_audit.py               # MANIFEST → cross-source byte-identical collision report
  smoke_test_yolo26.py         # YOLO26 install / NCNN export sanity check
  pi/                          # on-device shell helpers (Hailo benchmarks)

data/                          # gitignored. Datasets downloaded on demand.
  zenodo/                      # client-listed: 813 imgs, ripe/unripe/peduncle bboxes
  disease/                     # Kaggle Afzaal: 7 disease classes, LabelMe polygons
  osf-ej5qv/                   # OSF: 2967 classification images (some overlap with Kaggle)
  roboflow/
    afzaal-bbox-v4/            # bbox re-annotation of Kaggle Afzaal
    matt-lucky-ripeness/       # client-listed: ripe/unripe bboxes
    research-proj-disease/     # 10-class incl. native Healthy Fruit/Leaf/Flower
  strawdi/                     # StrawDI_Db1: 3100 instance-segmentation masks
                               # (non-commercial-academic license — flag before commercial deploy)

models/                        # gitignored. Trained weights + NCNN/HEF exports.
reports/                       # gitignored except dedup-report.md

agri-project-goals.docx        # original client spec (reference)
CLAUDE.md                      # agent guidance for this repo
```

Every dataset directory has a `LICENSE.md` recording its source license — needed for the eventual commercial-deploy audit.

## Phases (current status)

| # | Phase | Status |
|---|---|---|
| 1 | Data Prep & Scaffolding | **In progress** — skeleton + manifest library + per-source LICENSEs + `data/MANIFEST.json` (20,313 entries across 7 sources) done; dedup audit + disease crops outstanding |
| 2 | Detection Model — YOLO26n on Zenodo | Not started |
| 3 | Disease Classification — YOLO26n-cls + native healthy class | Not started |
| 4 | Integrated CPU Pipeline + Evaluator | Not started |
| 5 | Hailo Backend Port | Not started — but Hailo runtime + apps already validated on-device |

Phase 5 was originally the highest-risk phase (untested Hailo integration). After on-device validation in April 2026 it inverted: Hailo is the earliest-proven piece. Upstream training and integration are now the dominant risks.

## Setup

```bash
# Python 3.11 venv, deps pinned in requirements.txt
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Smoke test: load YOLO26n, export to NCNN, run on one image
.venv/bin/python scripts/smoke_test_yolo26.py
```

Datasets are not committed. Place each source under `data/<source>/` matching the layout in `.planning/research/2026-04-21-dataset-inventory-and-splits.md`.

### macOS dev gotcha

If `scripts/build_manifest.py` hangs at 0% CPU on first run, macOS Spotlight + `mediaanalysisd` are computing photo embeddings on the dataset images and serializing reads at the kernel level. Add `data/` to **System Settings → Siri & Spotlight → Spotlight Privacy**, or run `sudo mdutil -i off /System/Volumes/Data` (reversible). Builds are instant once indexing is excluded.

## Workflow

This project uses [GSD](https://github.com/karpathy/get-shit-done-style-planning) for phase-based planning. The high-traffic commands:

```bash
/gsd-progress          # where am I
/gsd-next              # advance to next logical step
/gsd-plan-phase N      # decompose phase N into plans
/gsd-execute-phase N   # run all plans in phase N
```

Decisions, scope, and progress live in `.planning/`. If a planning doc and the code disagree, the planning doc is wrong — fix it.

## Key technical decisions

Captured in full in `.planning/PROJECT.md`. Highlights:

- **YOLO26 over YOLOv12** — Ultralytics flags v12 as research-only (training instability). YOLO26 shipped 2026-01-14.
- **Two-stage pipeline, not unified seg** — datasets have no image overlap, so a single seg model isn't trainable from them.
- **Native `healthy` class from Roboflow research-proj-disease** — was originally going to synthesize healthy crops; not needed.
- **Hailo-10H, not Hailo-8** — corrected after on-device inspection (April 2026); planning docs preserve the superseded decision.
- **NCNN FP32 only, no INT8** — INT8 on NCNN isn't viable on Pi 5 yet.
- **Detection target**: ripe / unripe / peduncle. **Disease target**: 7 fruit-disease classes + healthy (top-1 ≥ 90 %). Peduncle inclusion and disease-class shape are open scope questions for the client.

## Status snapshot

Phase 1 dataset acquisition complete. All 7 sources downloaded on the Pi (zenodo, kaggle_afzaal, osf_ej5qv, roboflow ×3, strawdi); `data/MANIFEST.json` populated end-to-end at 20,313 entries. Remaining Phase 1 deliverables: cross-source dedup audit and per-polygon disease crop generation. See `.planning/STATE.md` for the live picture.

## License

Source code is MIT licensed (see [`LICENSE`](LICENSE)).

Datasets under `data/` and any model weights derived from them are governed by their own upstream licenses — see each `data/<source>/LICENSE.md`. The MIT grant covers this repo's source only; it does not relicense third-party data or models.

## Acknowledgements

- Strawberry detection dataset from Zenodo (Afzaal et al.) — see `data/zenodo/LICENSE.md`.
- Strawberry disease classification dataset from Kaggle — see `data/disease/LICENSE.md`.
- Baseline reference: BrunoKreiner's YOLOv8-XL instance-segmentation work on the Kaggle disease dataset (~92–93 % mAP50, 2023).
- Built with [Ultralytics YOLO26](https://docs.ultralytics.com), [NCNN](https://github.com/Tencent/ncnn), and [Hailo Dataflow Compiler](https://hailo.ai/developer-zone/).

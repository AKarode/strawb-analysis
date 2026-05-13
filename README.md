# Strawberry Vision Pi

Offline batch image-analysis pipeline for strawberry field photos, deployed on a Raspberry Pi 5 with the AI HAT+ 2 (Hailo-10H, 40 TOPS).

Given a folder of 500-1,000 RGB field images, it emits a single CSV with per-image counts (ripe / unripe / peduncle), per-fruit disease predictions, and stage timings — plus an offline evaluator that compares the CSV against ground-truth annotations to report detection mAP and per-image count MAE/Pearson.

**Not** real-time / streaming. **Not** multispectral (the legacy `multi-spectral/` folder name is a misnomer — input is plain RGB). **Not** an LLM project.

## Measured results (Pi 5, CPU-only, NCNN FP32)

| | |
|---|---|
| 1,000-image batch wall-clock | **5 min 13 sec** (target was 30 min — 5.8× under budget) |
| Per-image end-to-end | 308 ms mean / 235 ms p50 / 618 ms p95 |
| Ripe-fruit detection mAP50 | **0.917** |
| Disease classification top-1 (8 classes) | **0.939** (top-5 0.999) |
| Ripe-count MAE vs ground truth | **0.28 fruit / image** (77% exact-match, Pearson 0.89) |

The full one-pager — including per-class breakdowns, calibration guidance, and thermal notes — is at [`reports/anand_summary.md`](reports/anand_summary.md). Ranked improvement levers (S/M/L effort + expected lift) at [`reports/improvement_plan.md`](reports/improvement_plan.md).

The Hailo accelerator on the AI HAT+ 2 is **not** used in these numbers — CPU baseline alone already over-delivers vs the contract budget. Hailo path is the future 5-10× speed-up (Phase 5, deferred).

## How it works

Two-stage detect-then-classify pipeline:

```
images/  →  YOLO26n (detect)  →  crop each fruit  →  YOLO26n-cls (disease)  →  per-image CSV
                                                                                  │
                                                                                  ▼
                                                                          evaluator → mAP, count MAE/Pearson
```

Detector and classifier are trained separately because the source datasets cannot be merged into one segmentation model — they have no image overlap and incompatible label schemas (Zenodo has bbox for ripe/unripe/peduncle; Kaggle Afzaal has polygons for disease only).

Two backends share one entry point (`src/run_inference.py`):

| Backend | Runtime | Status |
|---|---|---|
| `cpu` | NCNN FP32 @ 640×640 (detect) + 224×224 (cls) | **Production-ready.** 5 min 13 sec for 1000 images on Pi 5. |
| `hailo` | Hailo HEF on AI HAT+ 2 | **Deferred** (Phase 5). CPU budget already met with 5.8× margin. |

INT8 on NCNN is **not** viable on Pi 5 as of April 2026 — CPU path stays FP32.

## Hardware

- Raspberry Pi 5 Model B Rev 1.1, 16 GB RAM
- AI HAT+ 2 with Hailo-10H (40 TOPS) — present, not yet integrated
- Access via Raspberry Pi Connect (no SSH)

Thermal during the 1000-image run: 64°C → 73°C, no active throttling. For sustained back-to-back batches, a passive heatsink or 30 mm fan on the AI HAT+ 2 is recommended.

## Repository layout

```
.planning/                     # GSD planning workflow — source of truth for decisions
  PROJECT.md                   # context, core value, constraints, key decisions
  REQUIREMENTS.md              # 25 v1 requirements across DATA / DETECT / DISEASE / PIPE / HAILO / EVAL
  ROADMAP.md                   # 5-phase plan
  STATE.md                     # live phase + progress snapshot
  research/                    # verified technology landscape (datasets, YOLO26, Hailo)

src/                           # library code
  manifest/                    # per-image provenance: schema, hashing, dataset scanners
  run_inference.py             # Phase 4 integrated CPU pipeline (--backend cpu|hailo)
  evaluator.py                 # Phase 4 count-accuracy comparison vs Zenodo ground truth

scripts/                       # CLIs (training, export, benchmark, eval)
  build_manifest.py            # walks data/, emits data/MANIFEST.json
  build_disease_crops.py       # cuts polygon/bbox crops into data/disease_crops/<split>/<class>/
  train_detect.py              # YOLO26n detector trainer
  train_disease.py             # YOLO26n-cls disease trainer
  export_detect_ncnn.py        # Mac-side NCNN export for detector
  export_cls_ncnn.py           # Mac-side NCNN export for classifier
  bench_detect_ncnn.py         # Pi-side detector latency bench
  bench_cls_ncnn.py            # Pi-side classifier latency + accuracy bench
  eval_disease_test.py         # Mac/Pi held-out cls test evaluator (per-image CSV + summary JSON)
  smoke_test_yolo26.py         # YOLO26 install / NCNN export sanity check

notebooks/
  train_detect_colab.ipynb     # Colab flow for detector (Zenodo)
  train_disease_colab.ipynb    # Colab flow for classifier (Kaggle Afzaal + Roboflow healthy)

docs/                          # Cursor megaprompts + session logs
  cursor-bootstrap-prompt.md   # Pi-side Cursor setup
  cursor-phase2-bench.md       # Pi detector bench megaprompt
  cursor-phase3-cls-bench.md   # Pi cls bench + integrated E2E megaprompt
  cursor-phase3-1k-batch.md    # Pi real 1000-image batch megaprompt
  cursor-phase4-eval.md        # Pi count evaluator megaprompt
  SESSION-*.md                 # session logs

data/                          # gitignored. Datasets downloaded on demand.
  zenodo/                      # 813 imgs, ripe/unripe/peduncle bboxes (CC-BY 4.0)
  disease/                     # Kaggle Afzaal: 7 disease classes, LabelMe polygons (3,200 src imgs)
  roboflow/
    research-proj-disease/     # 10-class; we use the native Healthy Fruit annotations
    afzaal-bbox-v4/            # bbox re-annotation of Kaggle Afzaal (unused in v1)
    matt-lucky-ripeness/       # additional ripe/unripe field images (unused in v1)
  osf-ej5qv/                   # OSF: classification subdirs; flagged DO NOT TRAIN (provenance unclear)
  strawdi/                     # 3,100 instance masks; **non-commercial-academic** — exclude from prod weights

models/                        # gitignored. Trained weights + NCNN exports.
  detect/yolo26n_zenodo.pt + yolo26n_zenodo_ncnn_model/
  disease/yolo26n_cls.pt + yolo26n_cls_ncnn_model/

reports/                       # gitignored except .md docs (anand_summary, improvement_plan)
```

Every dataset directory has a `LICENSE.md` recording its source license — needed for the commercial-deploy audit.

## Phases (current status)

| # | Phase | Status |
|---|---|---|
| 1 | Data Prep & Scaffolding | **~70%** — manifest + per-source LICENSEs + `data/MANIFEST.json` (20,313 entries) done. Cross-source dedup audit outstanding (non-blocking). |
| 2 | Detection Model — YOLO26n on Zenodo | **✅** Trained, NCNN-exported, Pi-validated. Release [`v0.2.0-detect`](https://github.com/AKarode/strawb-analysis/releases/tag/v0.2.0-detect). |
| 3 | Disease Classification — YOLO26n-cls (8 classes) | **✅** Trained, NCNN-exported, test-verified, Pi-benched. Release [`v0.3.0-disease`](https://github.com/AKarode/strawb-analysis/releases/tag/v0.3.0-disease). |
| 4 | Integrated CPU Pipeline + Evaluator | **✅** Pipeline runs end-to-end on Pi; count evaluator vs Zenodo ground truth produced. |
| 5 | Hailo Backend Port | **⏸ Deferred** — CPU baseline 5.8× under budget. Hailo is the future speed-up, not blocking. |

## Quick start

### Run inference on a folder of images (Pi-side)

```bash
# After pulling release v0.2.0-detect + v0.3.0-disease bundles
python -m src.run_inference \
  --backend cpu \
  --images <path-to-folder-of-jpgs> \
  --detector models/detect/yolo26n_zenodo_ncnn_model \
  --classifier models/disease/yolo26n_cls_ncnn_model \
  --out reports/inference.csv
```

Output: one CSV row per image with `n_ripe, n_unripe, n_peduncle, diseases_present, detect_ms, classify_ms, total_ms` etc.

### Compare CSV to ground truth (Zenodo schema)

```bash
python -m src.evaluator \
  --predictions reports/inference.csv \
  --ground-truth-images data/zenodo/strawberries/validation \
  --out reports/eval.json
```

Output: count MAE / exact-match-rate / Pearson for `n_total`, `n_ripe`, `n_unripe`.

### Reproduce the Pi bench from scratch

Three single-paste Cursor megaprompts close the loop, in order:

1. [`docs/cursor-phase3-cls-bench.md`](docs/cursor-phase3-cls-bench.md) — pulls release, rebuilds crops, runs cls bench + integrated E2E on Zenodo val.
2. [`docs/cursor-phase3-1k-batch.md`](docs/cursor-phase3-1k-batch.md) — assembles real 1,000-image batch and runs the full pipeline.
3. [`docs/cursor-phase4-eval.md`](docs/cursor-phase4-eval.md) — runs count evaluator vs ground truth.

### Setup (Mac-side dev)

```bash
# uv-managed env (recommended) — anaconda3 torch is broken on this dev machine
uv run --python 3.11 --with "ultralytics==8.4.48" --with torch --with torchvision --with pillow \
  python scripts/smoke_test_yolo26.py
```

Datasets are not committed. Place each source under `data/<source>/` matching the layout in `.planning/research/2026-04-21-dataset-inventory-and-splits.md`.

### macOS dev gotcha

If `scripts/build_manifest.py` hangs at 0% CPU on first run, macOS Spotlight + `mediaanalysisd` are computing photo embeddings on the dataset images and serializing reads at the kernel level. Add `data/` to **System Settings → Siri & Spotlight → Spotlight Privacy**, or run `sudo mdutil -i off /System/Volumes/Data` (reversible).

## Workflow

Three-machine workflow:

```
Mac dev workstation                  Colab (Pro)                Raspberry Pi 5 + AI HAT+ 2
─────────────────────                ────────────               ───────────────────────────
authors all code                     trains weights             executes inference + benchmarks
                                                                hosts raw datasets
   │                                    │   ▲                       │   ▲
   │  git push (PR-then-merge)          │   │ git clone + curl       │   │ git pull + gh release download
   ▼                                    ▼   │   Zenodo               ▼   │
GitHub (public): AKarode/strawb-analysis ←──┴────────────────────────┘
                                            │
                            best.pt ────────┘ (via Drive mount → gh release)
```

Cursor on the Pi (Raspberry Pi Connect — no SSH) executes Mac-authored code via paste-ready megaprompts. The Mac side does **not** run training; the Pi side does **not** author code or commits. Anand-facing client docs (see `reports/`) live in this repo so the audit trail and the data are co-located.

## Key technical decisions

Captured in full in `.planning/PROJECT.md`. Highlights:

- **YOLO26 over YOLOv12** — Ultralytics flags v12 as research-only. YOLO26 shipped 2026-01-14.
- **Two-stage pipeline, not unified seg** — datasets have no image overlap.
- **n-size models for both detector and classifier** — Phase 2 ablation showed yolo26s gives only +0.0022 mAP50-95 over n; not worth the Pi-side latency cost. yolo26n-cls trained directly without ablation given it already hit 98.9% val top-1.
- **NCNN FP32, no INT8** — INT8 on NCNN isn't viable on Pi 5 as of April 2026.
- **Hailo-10H, not Hailo-8** — corrected after on-device inspection (April 2026); planning docs preserve the superseded decision.
- **Native `healthy` class from Roboflow research-proj-disease** — cleaner than synthesizing from Zenodo non-diseased fruits.
- **Releases ship weights, not crops** — Kaggle Afzaal + Roboflow research-proj-disease redistribution licensing is unclear. Pi rebuilds crops locally via `scripts/build_disease_crops.py`.

## License

Source code is MIT licensed (see [`LICENSE`](LICENSE)).

Datasets under `data/` and model weights derived from them are governed by their own upstream licenses — see each `data/<source>/LICENSE.md`. The MIT grant covers this repo's source only; it does not relicense third-party data or models.

## Acknowledgements

- Strawberry detection dataset from Zenodo (Pastell et al., record 6126677) — CC-BY 4.0.
- Strawberry disease classification dataset from Kaggle (Usman Afzaal) — see `data/disease/LICENSE.md`.
- Roboflow research-proj/strawberry-diseases-detection for native healthy crops.
- Baseline reference: BrunoKreiner's YOLOv8-XL instance-segmentation work on the Kaggle disease dataset (~92-93% mAP50, 2023).
- Built with [Ultralytics YOLO26](https://docs.ultralytics.com), [NCNN](https://github.com/Tencent/ncnn), and [Hailo Dataflow Compiler](https://hailo.ai/developer-zone/).

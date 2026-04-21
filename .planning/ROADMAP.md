# Roadmap: Strawberry Vision Pi

**Defined:** 2026-04-20
**Granularity:** coarse
**Total Phases:** 5
**Total v1 Requirements:** 25
**Coverage:** 25/25 (100%)

## Core Value (from PROJECT.md)

A single-command batch inference tool that processes 500–1000 field images on a Raspberry Pi in well under the 30-minute budget, with a CSV that lets the client directly compare model predictions against annotated ground truth on the exact datasets they supplied.

## Phases

- [ ] **Phase 1: Data Prep & Scaffolding** — Project skeleton in place; both datasets converted to training-ready form with a synthesized `healthy` class.
- [ ] **Phase 2: Detection Model** — YOLO26n detector trained on Zenodo (ripe/unripe/peduncle), exported for CPU, evaluated on a held-out split.
- [ ] **Phase 3: Disease Classification Model** — YOLO26n-cls trained on disease crops (7 classes + synthesized `healthy`), exported for CPU, meets the 90%+ benchmark.
- [ ] **Phase 4: Integrated CPU Pipeline + Evaluator** — Single-command two-stage NCNN pipeline emits per-image CSV; offline evaluator reports mAP and per-class disease accuracy against ground truth.
- [ ] **Phase 5: Hailo Backend Port** — Same `run_inference.py` works on Pi 5 + AI HAT+ via HEF, with a side-by-side CPU vs Hailo timing report.

## Phase Details

### Phase 1: Data Prep & Scaffolding

**Goal**: Both provided datasets are organized into a reproducible, training-ready layout with a clear scaffold everything else can build on.
**Depends on**: Nothing (first phase)
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04
**Success Criteria** (what must be TRUE):
  1. The Zenodo detection dataset is laid out under `data/zenodo/` in stock YOLO format with train/val splits from source preserved, and an Ultralytics `data.yaml` points at those splits.
  2. The Kaggle disease dataset is converted from LabelMe polygons into per-polygon padded crops organized under `data/disease_crops/{train,val,test}/<class>/` with the source splits preserved.
  3. A `healthy` 8th class exists under `data/disease_crops/{train,val,test}/healthy/` populated by deterministic sampling of non-diseased fruit crops from Zenodo.
  4. Re-running the conversion scripts on the same inputs produces byte-identical outputs (deterministic, fixed seed on any sampling step).
**Plans**: TBD

### Phase 2: Detection Model

**Goal**: A trained, exported, evaluable fruit/peduncle detector ready to drop into the CPU pipeline.
**Depends on**: Phase 1
**Requirements**: DETECT-01, DETECT-02, DETECT-03, DETECT-04
**Success Criteria** (what must be TRUE):
  1. YOLO26n has been trained to completion on Zenodo with classes `ripe`, `unripe`, `peduncle` and a final `best.pt` is saved under `models/detect/`.
  2. Alongside `best.pt`, exported ONNX and NCNN artifacts for the detector are present under `models/detect/` and loadable by their respective runtimes.
  3. A training report captures validation-set mAP@50 and mAP@50-95 for the detector on the held-out Zenodo val split.
  4. Running a standalone detection inference script against a folder of images produces per-image bounding-box outputs the user can inspect.
**Plans**: TBD

### Phase 3: Disease Classification Model

**Goal**: A trained disease classifier that matches or beats the published baseline and has been sanity-checked on real-pipeline crops.
**Depends on**: Phase 1 (crop dataset), Phase 2 (Zenodo crops for sanity check)
**Requirements**: DISEASE-01, DISEASE-02, DISEASE-03, DISEASE-04
**Success Criteria** (what must be TRUE):
  1. YOLO26n-cls has been trained to completion on the prepared disease crop dataset covering 8 classes (7 Kaggle disease classes + synthesized `healthy`).
  2. Alongside the classifier `best.pt`, exported ONNX and NCNN artifacts for the classifier are present under `models/disease/` and loadable by their respective runtimes.
  3. A training report captures test-set per-class accuracy and overall top-1 accuracy; top-1 on the Kaggle held-out test split meets or beats 90%.
  4. A cross-dataset sanity report shows the classifier's predicted class distribution when run over Phase 2 Zenodo-sourced fruit crops, flagging degenerate behaviour (e.g., collapse to a single class) if present.
**Plans**: TBD

### Phase 4: Integrated CPU Pipeline + Evaluator

**Goal**: A single command processes N images end-to-end on Pi 5 CPU, emits the client-facing CSV, and a companion evaluator turns that CSV into ground-truth accuracy metrics.
**Depends on**: Phase 2 (detector NCNN artifact), Phase 3 (classifier NCNN artifact)
**Requirements**: PIPE-01, PIPE-02, PIPE-03, PIPE-04, PIPE-05, PIPE-06, EVAL-01, EVAL-02, EVAL-03
**Success Criteria** (what must be TRUE):
  1. Running `run_inference.py --images <dir> --count N --backend cpu` processes exactly N images through the two-stage detect→crop→classify pipeline and writes one CSV row per image with image id, total/ripe/unripe/peduncle counts, per-fruit disease predictions, stage timings, and total inference time.
  2. A warm-up pass runs before the first timed inference, and the run prints end-of-run summary statistics (mean, p50, p95 inference time; total runtime; images processed).
  3. Re-running the same command with the same inputs and weights produces an identical CSV (modulo timing columns).
  4. Running `evaluate.py` against the produced CSV plus the dataset ground-truth annotations emits a single Markdown/plain-text report containing detection mAP@50, mAP@50-95, and per-class disease accuracy over whichever images have matching ground truth.
**Plans**: TBD

### Phase 5: Hailo Backend Port

**Goal**: The same inference command runs on Pi 5 + AI HAT+ via a Hailo HEF backend, with measurable CPU-vs-Hailo throughput evidence.
**Depends on**: Phase 4 (working CPU pipeline + CSV schema)
**Requirements**: HAILO-01, HAILO-02, HAILO-03, HAILO-04
**Success Criteria** (what must be TRUE):
  1. A `HailoBackend` class runs the detection model as a Hailo HEF file on Pi 5 + AI HAT+ (Hailo-8, 26 TOPS).
  2. Running `run_inference.py --backend hailo` on the Pi 5 produces CSV output in exactly the same schema as the CPU backend; for the disease classifier, the Hailo path either runs as HEF or falls back to NCNN with the fallback clearly documented per run.
  3. A side-by-side report documents CPU vs Hailo per-image inference time over at least 100 representative images (means, p50, p95).
  4. The Hailo run on Pi 5 still completes N=1000 images inside the 30-minute soft ceiling.
**Plans**: TBD
**Research flag**: yes — `/gsd-research-phase` should confirm YOLO26 Hailo Model Zoo official support status (April 2026 ship target per Ultralytics discussion #23655) before planning. If official support has landed, target YOLO26n HEF; otherwise use YOLO11n HEF as the interim detection backend per the Phase 5 stopgap decision.

## Dependency Graph

```
Phase 1 (Data Prep & Scaffolding)
   │
   ├──> Phase 2 (Detection Model)
   │       │
   │       └──> Phase 3 (Disease Classifier) ──┐
   │                                           │
   │                                           ▼
   │                                       Phase 4 (CPU Pipeline + Eval)
   │                                           │
   └───────────────────────────────────────────┘
                                               │
                                               ▼
                                           Phase 5 (Hailo Port)  [research flag]
```

Phase 1 blocks everything. Phases 2 and 3 could in principle parallelize but are kept sequential (Phase 3 depends on Phase 2's Zenodo crops for the cross-dataset sanity check). Phase 4 integrates both models. Phase 5 is a pure backend swap on top of Phase 4.

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Data Prep & Scaffolding | 0/0 | Not started | — |
| 2. Detection Model | 0/0 | Not started | — |
| 3. Disease Classification Model | 0/0 | Not started | — |
| 4. Integrated CPU Pipeline + Evaluator | 0/0 | Not started | — |
| 5. Hailo Backend Port | 0/0 | Not started | — |

## Coverage Validation

All 25 v1 requirements are mapped to exactly one phase. No orphans, no duplicates.

| Phase | Requirements | Count |
|-------|--------------|-------|
| 1 | DATA-01, DATA-02, DATA-03, DATA-04 | 4 |
| 2 | DETECT-01, DETECT-02, DETECT-03, DETECT-04 | 4 |
| 3 | DISEASE-01, DISEASE-02, DISEASE-03, DISEASE-04 | 4 |
| 4 | PIPE-01, PIPE-02, PIPE-03, PIPE-04, PIPE-05, PIPE-06, EVAL-01, EVAL-02, EVAL-03 | 9 |
| 5 | HAILO-01, HAILO-02, HAILO-03, HAILO-04 | 4 |
| **Total** | | **25 / 25** |

---
*Roadmap created: 2026-04-20 (via /gsd-new-project roadmapper)*

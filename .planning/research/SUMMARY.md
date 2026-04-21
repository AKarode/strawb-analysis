# Project Research Summary

**Project:** Strawberry Vision Pi — Edge Inference Pipeline
**Domain:** Edge computer vision (agricultural, offline batch)
**Researched:** 2026-04-20
**Confidence:** HIGH
**Method:** Research conducted interactively in-session — official docs, GitHub source verification, Gemini 3 cross-check, live benchmark lookups. Claims verified against primary sources before acceptance.

## Executive Summary

This is an offline edge computer-vision inference project — not an AI/ML research project and not an LLM project. The client asked for YOLO-based batch inference of 500–1000 strawberry field images on a Raspberry Pi 5, with and without the Hailo AI HAT+. Scope is well-bounded: four deterministic goals (presence, count, ripe/unripe, disease) over provided datasets, producing CSV output with ground-truth comparison.

The recommended approach is a **two-stage pipeline using Ultralytics YOLO26 (released Jan 14, 2026)**: a detection model trained on the Zenodo dataset (813 images, classes ripe/unripe/peduncle) provides fruit localization and counting; a classification model trained on per-polygon crops from the Kaggle disease dataset (~2500 images, 7 classes) labels each detected fruit. Two inference backends are required deliverables — Pi 5 CPU via NCNN (YOLO26n FP32, 67.69 ms/image verified benchmark) and Pi 5 + Hailo AI HAT+ Hailo-8 via HEF.

Primary risks: (1) the Kaggle disease dataset is curated 419×419 close-ups rather than field-scene images, so the disease model's real-world transferability is uncertain even though intra-dataset mAP reaches 92–93% (BrunoKreiner's published YOLOv8 baseline on the same data); (2) YOLO26 official Hailo support is arriving in April 2026 — stopgap for the Hailo backend is YOLO11n (last Hailo-Model-Zoo-supported version) until YOLO26 HEF lands.

## Key Findings

### Recommended Stack

Ultralytics + NCNN on the CPU path, Ultralytics + Hailo Dataflow Compiler on the accelerator path. Python-only project. No LLM. No multispectral processing.

**Core technologies:**
- **Ultralytics YOLO26n** (detection) — Jan 14, 2026 release, Ultralytics' edge-optimized production flagship. NMS-free, DFL-free, explicitly designed for CPU edge deployment. Official benchmark: **67.69 ms/image** NCNN FP32 @ 640×640 on Raspberry Pi 5. Replaces YOLOv12 which Ultralytics itself marks as "benchmarking/research only" due to training instability.
- **Ultralytics YOLO26n-cls** (classification) — same family, 224×224 input, 5.0 ms ONNX x86 / ~15 ms projected Pi 5 NCNN. Used as the second stage for disease.
- **NCNN** (Tencent) — CPU inference runtime for Pi 5 ARM NEON. FP32 only; INT8 + NCNN is not viable on Pi 5 as of April 2026 per published benchmarks.
- **Hailo Dataflow Compiler + Model Zoo** — YOLO11n is the most recent Hailo-supported version for official conversion today. YOLO26 Hailo support is shipping this month (April 2026) per Ultralytics discussion #23655. Community pipeline at [DanielDubinsky/yolo26_hailo](https://github.com/DanielDubinsky/yolo26_hailo) can bridge if official timing slips.
- **Raspberry Pi 5 + AI HAT+ (Hailo-8, 26 TOPS)** — correct hardware SKU for this workload. NOT the AI HAT+ 2 (Hailo-10H, 40 TOPS, $130) — that targets on-device LLM/VLM.
- **Python 3.11+** with `ultralytics`, `onnx`, `ncnn`, `hailo-platform` (Hailo runtime), `opencv-python`, `pandas` (for CSV).

### Expected Features

Client docx + WhatsApp clarifications define the complete feature set. No "table stakes vs. differentiators" judgment call here — the spec is concrete.

**Must have (all explicit in client docx):**
- Detect strawberry presence/absence — Goal #1
- Count total strawberries per image — Goal #2
- Ripe vs. unripe classification — Goal #3
- Disease detection — Goal #4
- CPU-only inference mode on RPi — explicit
- Hailo AI HAT+ inference mode — explicit
- Script accepts N argument (500 or 1000) — explicit
- Per-image CSV with per-inference timings — explicit
- Ground-truth accuracy comparison — explicit

**Should have (quality-of-life):**
- Reproducible runs (fixed seed, deterministic preprocessing)
- Warm-up pass before first timed inference (otherwise model load dominates first-image timing)
- Separate timing columns for each pipeline stage (detection, crop, classification)
- Summary statistics (mean/p50/p95 ms, total runtime) printed at end

**Defer (explicitly out of scope):**
- Multispectral — not mentioned in docx
- Streaming / real-time — client confirmed batch-after-mapping
- Drone imagery — datasets don't support it
- v2 dataset (Roboflow) — can pull later if mAP is weak

### Architecture Approach

A two-stage pipeline per image: detection produces fruit bounding boxes (ripe/unripe/peduncle), then each fruit box is cropped and passed to a disease classifier. Both models share the same backend abstraction so the same driver script works on CPU (NCNN) and Hailo (HEF). CSV writer collects per-image rows with counts and disease predictions. A separate eval script reads the CSV plus ground-truth annotations and produces accuracy metrics (mAP for detection, per-class accuracy for disease).

**Major components:**
1. **Preprocess** — load image, letterbox to model input size, normalize. Same for both backends.
2. **Detector (YOLO26n)** — emits fruit/peduncle bboxes + class + confidence. Backend-swappable.
3. **Crop + post-filter** — crop each detected fruit box, skip peduncles, keep only ripe/unripe for disease stage.
4. **Classifier (YOLO26n-cls)** — emits disease class + confidence for each crop. Backend-swappable.
5. **CSV writer** — one row per input image; includes counts, per-fruit disease summary, per-stage timings.
6. **Evaluator** — offline script reading CSV + ground truth annotations; produces summary report.
7. **Backend adapters** — `NcnnBackend` (CPU) and `HailoBackend` (HEF) implementing a common `run(image)` interface so the pipeline code is identical.

### Critical Pitfalls

1. **Domain transfer for disease model.** Kaggle disease dataset is 419×419 curated close-ups; our pipeline crops from real field images (variable lighting, scale, occlusion). **Avoid**: test disease classifier on cropped-from-Zenodo fruits before claiming accuracy. Report confidence distribution to flag low-confidence predictions.
2. **Missing `healthy` class in disease dataset.** All 7 classes are diseases — there's no null class. A classifier trained only on disease classes will *always* predict disease even for healthy fruit. **Avoid**: synthesize a `healthy` class by sampling non-diseased fruit crops from Zenodo, or add a confidence threshold + "unknown" bucket.
3. **YOLO26 Hailo support timing.** Hailo Model Zoo doesn't support YOLO26 as of today; official support is "April 2026." **Avoid**: build Hailo backend against YOLO11n first (known working path), then swap to YOLO26 when official HEF conversion lands. Treat the Hailo YOLO26 migration as a discrete later phase.
4. **INT8 on Pi 5 + NCNN not viable.** Don't chase INT8 on CPU — benchmarks show precision support failures. FP32 NCNN is already 67 ms/image, well under budget. **Avoid**: wasting time on INT8 CPU quantization for this project.
5. **First-image timing inflation.** Model load time can dominate the first inference, skewing per-image averages. **Avoid**: run a warm-up inference on a dummy image before starting timed runs.
6. **YOLO label schema mismatch.** Zenodo is YOLO bbox, Kaggle is LabelMe polygon — they cannot be merged into one training set. **Avoid**: any "unified model" approach. Train two separate models.
7. **Annotation coordinate coverage.** Disease dataset polygons span disease tissue (partial-fruit regions), not whole-fruit bboxes. Conversion to classification crops needs to compute a padded bounding box around each polygon, not the polygon itself.

## Implications for Roadmap

Research suggests a **coarse 5-phase roadmap** aligned with the "coarse granularity" config. Phases are dependency-ordered: each depends only on its predecessors.

### Phase 1: Project scaffolding + dataset prep
**Rationale:** Needed before any training. Downstream phases all depend on a clean dataset layout, a reproducible training environment, and a clear output schema. No YOLO-specific work yet — just file structure.
**Delivers:** Python package skeleton (`src/`, `scripts/`, `data/`, `models/`, `reports/`); dataset conversion scripts (LabelMe polygon → per-crop classification dataset with synthesized `healthy` class); training config templates; CSV schema document; `.gitignore`, `pyproject.toml` / `requirements.txt`.
**Addresses:** Pitfalls #2 (healthy class), #6 (schema mismatch), #7 (polygon→crop conversion).

### Phase 2: Train + evaluate detection model
**Rationale:** Detection is upstream of disease classification in the pipeline, so it must work before disease can be tested meaningfully. YOLO26n training on 813 Zenodo images is small and fast on any modern GPU or Colab.
**Delivers:** Trained YOLO26n weights (`.pt`), exported ONNX + NCNN artifacts, Zenodo-val mAP evaluation, detection-only inference script.
**Uses:** Ultralytics, Zenodo dataset.

### Phase 3: Train + evaluate disease classifier
**Rationale:** Depends on Phase 1 crop-dataset prep and validates Phase 2 crops. Must match or beat BrunoKreiner's 92% mAP50 baseline to be worth shipping.
**Delivers:** Trained YOLO26n-cls weights, exported ONNX + NCNN, per-class accuracy report against Kaggle test set, intra-dataset + cross-dataset (Zenodo crops) sanity check.
**Addresses:** Pitfall #1 (domain transfer testing).

### Phase 4: Integrated CPU pipeline + CSV + evaluator
**Rationale:** First end-to-end deliverable. Hits the CPU-mode success criterion. This alone would satisfy most of the client's ask.
**Delivers:** `run_inference.py --images <dir> --count N --backend cpu`, two-stage NCNN pipeline, per-image CSV writer, offline `evaluate.py` that reads CSV + ground truth, timing summary output.
**Addresses:** Pitfall #5 (first-image warm-up), CSV schema from Phase 1.

### Phase 5: Hailo backend port
**Rationale:** Last because it depends on a working CPU pipeline. Starts with YOLO11n (Hailo-Model-Zoo-supported today) and has a migration path to YOLO26 when official Hailo support lands.
**Delivers:** `HailoBackend` class, YOLO11n HEF conversion (detection), YOLO26n-cls HEF conversion if available else fallback classifier, `run_inference.py --backend hailo` working on Pi 5 with AI HAT+, side-by-side CPU vs Hailo timing report.
**Addresses:** Pitfall #3 (Hailo version timing).

### Phase Ordering Rationale

- Phase 1 is shared scaffolding everything else needs (datasets, CSV schema, package structure).
- Phases 2 and 3 could in principle be parallelized, but they share dataset prep outputs; keeping them sequential avoids wasted rework if conversion scripts need adjustment.
- Phase 4 integrates the two models — can't happen until both exist.
- Phase 5 is a port of Phase 4 to a new backend — strictly depends on Phase 4.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 5:** Hailo Dataflow Compiler setup, YOLO26 conversion status, model-zoo workflow — worth a `/gsd-research-phase` spawn right before planning, especially to confirm YOLO26 Hailo support has shipped by the time we get here.

Phases with standard patterns (can skip phase-level research):
- **Phase 1:** Standard Python packaging + data-prep work.
- **Phase 2:** Ultralytics YOLO26 training is extensively documented.
- **Phase 3:** Same as Phase 2.
- **Phase 4:** Standard Python glue code.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | YOLO26 release date, Pi 5 NCNN benchmarks, Hailo status all verified against Ultralytics + Raspberry Pi + GitHub primary sources. |
| Features | HIGH | Spec is concrete from client docx + WhatsApp confirmation. |
| Architecture | HIGH | Two-stage pipeline is forced by the shape of the datasets; alternatives (unified seg) were evaluated and rejected. |
| Pitfalls | MEDIUM | Domain-transfer concern is real but not quantified until we test — that's Phase 3's job. |

**Overall confidence:** HIGH

### Gaps to Address

- **Disease model real-world accuracy** — Can only be measured after Phase 3 when we run the classifier on Zenodo fruit crops. May require retraining with Zenodo-sourced crops if intra-dataset mAP doesn't transfer.
- **Hailo YOLO26 official support shipping timing** — Project-external dependency. Handle in Phase 5 via YOLO11n stopgap.
- **Client's actual capture rig / real deployment imagery** — Client said "same data format as in the dataset" for PoC. Real drone / rig imagery is a future phase, explicitly out of scope here.

## Sources

### Primary (HIGH confidence)
- [Ultralytics YOLO26 docs](https://docs.ultralytics.com/models/yolo26/) — release date Jan 14, 2026; model sizes + COCO benchmarks
- [Ultralytics Raspberry Pi guide](https://docs.ultralytics.com/guides/raspberry-pi/) — YOLO26n Pi 5 NCNN 67.69 ms/image verified
- [Ultralytics discussion #23655](https://github.com/orgs/ultralytics/discussions/23655) — YOLO26 + Hailo-8L deployment pipeline and status
- [Hailo Model Zoo GitHub](https://github.com/hailo-ai/hailo_model_zoo) — confirms current YOLO11-obb as latest supported
- [Raspberry Pi AI HAT+ 2 announcement](https://www.raspberrypi.com/news/introducing-the-raspberry-pi-ai-hat-plus-2-generative-ai-on-raspberry-pi-5/) — Hailo-10H SKU for LLM workloads (not our use case)
- [BrunoKreiner/strawberry_diseases](https://github.com/BrunoKreiner/strawberry_diseases) — verified real, YOLOv8-XL segmentation, 92-93% mAP50 baseline on same Kaggle dataset
- [Zenodo strawberries dataset (record 6126677)](https://zenodo.org/record/6126677) — inspected: 813 images, 3 classes, YOLO format
- [Kaggle strawberry-disease-detection-dataset](https://www.kaggle.com/datasets/usmanafzaal/strawberry-disease-detection-dataset) — inspected: ~2500 images, 7 classes, LabelMe polygons, 419×419

### Secondary (MEDIUM confidence)
- [arXiv 2502.15737](https://arxiv.org/pdf/2502.15737) — YOLO performance analysis on edge devices, cited for Pi 5 INT8 infeasibility
- [DanielDubinsky/yolo26_hailo](https://github.com/DanielDubinsky/yolo26_hailo) — community YOLO26→HEF pipeline (fallback if official support slips)
- [CNX AI HAT+ 2 review](https://www.cnx-software.com/2026/01/20/raspberry-pi-ai-hat-2-review-a-40-tops-ai-accelerator-tested-with-computer-vision-llm-and-vlm-workloads/) — independent benchmarks on the LLM-focused SKU

### Tertiary (LOW confidence)
- Gemini 3 research brief (2026-04-20) — initial survey; individual claims re-verified against primary sources before inclusion here. Two of Gemini's claims were corrected (YOLO12 Hailo support status, INT8 NCNN feasibility).

---
*Research completed: 2026-04-20*
*Ready for roadmap: yes*

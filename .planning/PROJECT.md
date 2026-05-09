# Strawberry Vision Pi — Edge Inference Pipeline

## What This Is

An offline Raspberry Pi 5 computer-vision pipeline that batch-processes stored RGB strawberry field images and reports, per image: strawberry presence, count, ripeness (ripe/unripe), and disease classification. Runs in two modes — Pi 5 CPU-only and Pi 5 + Hailo AI HAT+ — emitting a CSV report with per-image inference timings and a ground-truth accuracy comparison against provided annotations. Built for a client exploring edge-deployable agricultural vision; first deliverable is a proof of concept against the two provided datasets.

## Core Value

A single-command batch inference tool that processes 500–1000 field images on a Raspberry Pi in well under the 30-minute budget, with a CSV that lets the client directly compare model predictions against annotated ground truth on the exact datasets they supplied. Everything else (multiple backends, Hailo acceleration, disease class breadth) serves this core value.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Detect strawberry presence/absence per image
- [ ] Count total strawberries per image
- [ ] Classify ripeness (ripe vs unripe) per detected strawberry
- [ ] Detect disease (7 classes from Kaggle dataset) per detected fruit
- [ ] Run on Raspberry Pi 5 CPU-only (no accelerator)
- [ ] Run on Raspberry Pi 5 + Hailo AI HAT+ (Hailo-8, 26 TOPS)
- [ ] Single-script batch runner: user specifies N (e.g. 500 or 1000), script processes folder of images
- [ ] Per-image CSV output (one row per image) including detection counts, ripeness breakdown, disease predictions, and inference time
- [ ] Ground-truth accuracy comparison against dataset annotations

### Out of Scope

- **Multispectral imaging** — client docx never mentions it; the "multi-spectral" folder name is stale. Scope is plain RGB.
- **Real-time streaming inference** — client confirmed via WhatsApp that mapping completes before processing starts. Assume all images are stored.
- **Drone / aerial imagery** — provided datasets are ground-level. Field drone deployment would require new data collection and is a future phase.
- **Leaf-only disease detection in full field scenes** — disease dataset is 419×419 curated close-ups, not field scenes. Leaf diseases only work when applied to cropped regions.
- **AI HAT+ 2 (Hailo-10H, 40 TOPS)** — targets LLM/VLM workloads. CV pipeline doesn't benefit from the extra cost.
- **Custom detection architecture** — using YOLO26 (official Ultralytics) for CPU; YOLO11n as interim Hailo backend. No research-grade architecture work.
- **INT8 quantization on Pi 5 + NCNN** — precision support not viable as of April 2026 per published benchmarks. FP32 NCNN is fast enough.

## Context

**Datasets (already pulled and inspected locally under `data/`):**
- **Zenodo strawberry detection** (813 imgs, YOLO-format, 3 classes: `ripe`/`unripe`/`peduncle`) — built originally for a harvesting-robot edge vision system. Ground-level canopy shots, variable resolution (640×480 through 4000×3000).
- **Kaggle disease detection** (Usman Afzaal, ~2500 imgs, LabelMe polygon, 7 classes) — all 419×419 pre-cropped close-ups of diseased leaves/fruits. Heavy class imbalance: 5 of 7 classes are leaf diseases, only 2 are fruit diseases (`anthracnose_fruit_rot`, `powdery_mildew_fruit`).
- **Third option mentioned (not yet pulled)**: Roboflow `matt-lucky-f7mch/strawberry-k4gtp`.

**Prior art / baseline:**
- [BrunoKreiner/strawberry_diseases](https://github.com/BrunoKreiner/strawberry_diseases) — verified real repo, last push July 2023, YOLOv8-XL instance segmentation on the exact same Kaggle dataset, reported **92–93% mAP50**. That's the accuracy benchmark our disease model must match or beat.

**Technology landscape (April 2026):**
- **YOLO26** released Jan 14, 2026 — Ultralytics' current edge-optimized production flagship. No DFL, no NMS, native end-to-end. Official Ultralytics recommendation for edge/CPU over YOLOv12 (which has training instability and slower CPU throughput).
- **YOLO11** is the most recent version with full Hailo Model Zoo support; YOLO26 Hailo official support is arriving April 2026 (per Ultralytics discussion #23655). Community pipeline exists at [DanielDubinsky/yolo26_hailo](https://github.com/DanielDubinsky/yolo26_hailo).
- **Pi 5 + NCNN benchmarks** — Ultralytics' published RPi guide reports 67.69 ms/image FP32 @ 640×640. **This number was not reproducible on our Pi 5 (May 2026)**: `yolo benchmark` with ultralytics 8.4.48 reports 183.73 ms/image NCNN inference time on yolo26n_zenodo. The model itself is healthy (mAP50 0.679, 4-thread NCNN with all FP16 + winograd + packing flags on, no throttling). Suspect causes of the gap: ultralytics version delta, RPi firmware, or Cursor agent (~25% CPU) running on the same box during measurement. Empirical operating budget: ~184 ms inference + ~200 ms decode of 8 MB / 4000×3000 source JPEGs = ~390 ms full predict pipeline per image. Detector throughput on 1000 images ≈ 6.5 min, well under the 30-min batch budget.

**Client input capture profile (from WhatsApp):**
- 500–1000 stored RGB images per run
- Images captured at 1–5 frames/sec during mapping, but batch inference starts only after mapping completes
- Client says: "same data format as the dataset" — no new collection required for PoC

## Constraints

- **Hardware**: Raspberry Pi 5 — must support both CPU-only path and Hailo AI HAT+ (Hailo-8, 26 TOPS) path
- **Throughput**: Process 500–1000 images within 30 minutes. Benchmarks suggest <5 min is achievable, so this is a soft ceiling not a tight constraint.
- **Framework**: YOLO family — client specified "v12 or v26." Decision: YOLO26 for CPU, YOLO11n for Hailo backend (until YOLO26 Hailo support lands).
- **Data format**: Use the datasets' native formats for PoC (Zenodo YOLO `.txt`, Kaggle LabelMe `.json`).
- **Offline**: No network dependency during inference on the Pi.
- **Output**: CSV, one row per image.
- **Reproducibility**: Every run must be reproducible — same model weights + same images → same CSV (deterministic, fixed seeds where applicable).

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Two-stage pipeline: detect (YOLO26n) → classify disease (YOLO26n-cls) on cropped fruits | Disease dataset is 419×419 curated crops, not field scenes. A unified seg model with partial labels across two non-overlapping datasets fails. Two-stage matches the data we have. | — Pending (validate in Phase 2) |
| YOLO26 over YOLO12 | Ultralytics' own docs recommend v11 or v26 for production; v12 is "benchmarking and research only" (training instability, slower CPU). | — Pending |
| Original AI HAT+ (Hailo-8, 26 TOPS) over AI HAT+ 2 (Hailo-10H, 40 TOPS) | AI HAT+ 2 targets LLM/VLM workloads. CV pipeline gains nothing from the extra cost. | — Pending |
| NCNN FP32 on Pi 5 CPU (skip INT8) | INT8 quantization on Pi 5 + NCNN is not viable as of April 2026 per published benchmarks (precision support issues). FP32 hits 67 ms/image, already well inside budget. | — Pending |
| Interim Hailo backend: YOLO11n | Hailo Model Zoo tops out at YOLO11 today. YOLO26 Hailo support is April 2026 — target for migration once official. | ⚠️ Revisit when YOLO26 Hailo support ships |
| Disease model: classification (YOLO26n-cls) over segmentation (YOLO26n-seg) | Disease dataset is effectively a classification dataset in polygon clothing. Classification is 6× faster and simpler; polygons get converted to per-crop labels. | — Pending |
| Add a synthetic `healthy` class to disease training by sampling non-diseased Zenodo fruits | Disease dataset has no null class. Detection pipeline will produce fruit crops regardless of disease presence. Without a `healthy` class the classifier has no way to say "no disease." | — Pending |
| Not pulling the Roboflow dataset for v1 | Zenodo + Kaggle are sufficient for detection + disease. Roboflow is optional augmentation, can revisit if mAP is weak. | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-20 after initialization*

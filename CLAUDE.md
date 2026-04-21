# CLAUDE.md — Strawberry Vision Pi

Claude Code guidance for this project. Read `.planning/PROJECT.md` for full context.

## Project

**Strawberry Vision Pi** — offline Raspberry Pi 5 computer-vision pipeline that batch-processes 500–1000 stored RGB strawberry field images and emits a per-image CSV report (presence, count, ripeness, disease) plus a ground-truth accuracy comparison. Two required backends: Pi 5 CPU (NCNN) and Pi 5 + Hailo AI HAT+ (Hailo-8 HEF).

**Not** an LLM project. **Not** multispectral. **Not** streaming / real-time.

## GSD Workflow

This project uses the GSD planning workflow. Planning artifacts live under `.planning/`.

- `.planning/PROJECT.md` — project context, core value, constraints, key decisions
- `.planning/REQUIREMENTS.md` — 25 v1 requirements with REQ-IDs across DATA, DETECT, DISEASE, PIPE, HAILO, EVAL
- `.planning/ROADMAP.md` — 5-phase plan; **Phase 5 has a research flag** for Hailo YOLO26 status
- `.planning/STATE.md` — current phase / progress
- `.planning/research/SUMMARY.md` — verified technology landscape (YOLO26, Hailo, Pi 5 benchmarks)
- `.planning/config.json` — `mode: yolo, granularity: coarse, model_profile: quality, research/plan_check/verifier: all on`

**Phase sequence:**
1. Data Prep & Scaffolding (DATA-01..04)
2. Detection Model — YOLO26n on Zenodo (DETECT-01..04)
3. Disease Classification Model — YOLO26n-cls on Kaggle crops + synthesized `healthy` (DISEASE-01..04)
4. Integrated CPU Pipeline + Evaluator (PIPE-01..06, EVAL-01..03)
5. Hailo Backend Port (HAILO-01..04)

**Next command:** `/gsd-plan-phase 1`

## Key Technical Decisions (from PROJECT.md)

- **Framework**: Ultralytics YOLO26 (released Jan 14, 2026) — NOT YOLOv12 (Ultralytics flags v12 as research-only; training instability).
- **CPU runtime**: NCNN FP32 @ 640×640 — Pi 5 benchmark is 67.69 ms/image for YOLO26n. Do NOT attempt INT8 on NCNN — not viable on Pi 5 as of April 2026.
- **Pipeline**: Two-stage — detect (YOLO26n) → crop each fruit → classify disease (YOLO26n-cls). Datasets cannot be merged into a unified seg model (no image overlap, incompatible label schemas).
- **Disease class**: 7 Kaggle disease classes + synthesized `healthy` class (sampled from non-diseased Zenodo fruits). Classifier must have a null option.
- **Hardware**: Original Raspberry Pi AI HAT+ (Hailo-8, 26 TOPS). **Not** AI HAT+ 2 (Hailo-10H, 40 TOPS — LLM-targeted, irrelevant here).
- **Hailo stopgap**: YOLO26 Hailo official support is April 2026. If the detector's HEF conversion isn't ready, Phase 5 falls back to YOLO11n for the Hailo detection path (last Hailo Model Zoo-supported YOLO).
- **Baseline to beat**: BrunoKreiner's 92–93% mAP50 on the same Kaggle disease dataset (YOLOv8-XL instance segmentation, 2023). Our classifier target: ≥ 90% top-1.

## Repo Layout

```
data/              # Downloaded datasets (gitignored). Zenodo + Kaggle under data/{zenodo,disease}/
models/            # Trained weights + exports (gitignored). detect/ and disease/
scripts/           # Training + conversion scripts
src/               # Library code: backends, pipeline, evaluator
reports/           # CSV + training reports (gitignored)
.planning/         # GSD workflow artifacts (committed)
agri-project-goals.docx  # Original client spec (reference)
```

## Running

Training happens on dev workstation / Colab. Inference runs on Pi 5. See phase plans for specifics.

## Brand / Repo Context

This project sits under the parent `Pyranthus/products/` monorepo. See root `CLAUDE.md` for umbrella context. `multi-spectral/` folder name is legacy — actual scope is plain RGB.

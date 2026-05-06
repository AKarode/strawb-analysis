# CLAUDE.md — Strawberry Vision Pi

Claude Code guidance for this project. Read `.planning/PROJECT.md` for full context.

This repo is **dual-machine**. The Mac dev workstation (this side) authors code and exported model artifacts; a Raspberry Pi 5 with the **AI HAT+ 2 (Hailo-10H, 40 TOPS)** executes inference and benchmarks. Pi-side AI agents (Cursor on the Pi, accessed via Raspberry Pi Connect — no SSH) read `AGENTS.md` for their role boundary, setup steps, and command surface. Don't propose SSH-driven workflows from this side — Pi-side execution is owned by the Pi-side agent.

The repo is **public** on GitHub at `AKarode/strawb-analysis`. Pi-side clones use plain HTTPS — no deploy keys, no PATs.

> **Hardware note:** Earlier planning docs (`.planning/PROJECT.md`, `.planning/REQUIREMENTS.md`) say "Hailo-8 / original AI HAT+." This was superseded by on-device inspection in April 2026 — the actual hardware is Hailo-10H / AI HAT+ 2. The planning docs preserve the old decision intentionally as a record of the change. **`README.md` is the source of truth for hardware.** Toolchain implications: Hailo-10H uses a newer DFC and has a sparser model zoo as of 2026; the YOLO11n fallback that worked on Hailo-8 may not transfer cleanly.

## Project

**Strawberry Vision Pi** — offline Raspberry Pi 5 computer-vision pipeline that batch-processes 500–1000 stored RGB strawberry field images and emits a per-image CSV report (presence, count, ripeness, disease) plus a ground-truth accuracy comparison. Two required backends: Pi 5 CPU (NCNN) and Pi 5 + Hailo AI HAT+ 2 (Hailo-10H HEF).

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
3. Disease Classification Model — YOLO26n-cls on Kaggle crops + native `healthy_fruit` from Roboflow research-proj-disease (DISEASE-01..04)
4. Integrated CPU Pipeline + Evaluator (PIPE-01..06, EVAL-01..03)
5. Hailo Backend Port (HAILO-01..04)

**Current state (2026-05-05):** Phase 1 in progress. `data/MANIFEST.json` populated at 20,313 entries on the Pi; Phase 2 trainer + Colab notebook ready (`notebooks/train_detect_colab.ipynb`). Outstanding Phase 1 items: dedup audit, disease crops. See `.planning/STATE.md` for the live picture.

## Key Technical Decisions (from PROJECT.md)

- **Framework**: Ultralytics YOLO26 (released Jan 14, 2026) — NOT YOLOv12 (Ultralytics flags v12 as research-only; training instability).
- **CPU runtime**: NCNN FP32 @ 640×640 — Pi 5 benchmark is 67.69 ms/image for YOLO26n. Do NOT attempt INT8 on NCNN — not viable on Pi 5 as of April 2026.
- **Pipeline**: Two-stage — detect (YOLO26n) → crop each fruit → classify disease (YOLO26n-cls). Datasets cannot be merged into a unified seg model (no image overlap, incompatible label schemas).
- **Disease class**: 7 Kaggle disease classes + native `healthy_fruit` from Roboflow research-proj-disease (originally planned to synthesize healthy crops from Zenodo non-diseased fruits — superseded; native source is cleaner). Classifier must have a null option.
- **Hardware**: Raspberry Pi 5 + AI HAT+ 2 (Hailo-10H, 40 TOPS). Confirmed via on-device inspection April 2026 — supersedes earlier "Hailo-8" entries in `.planning/`. Access via Raspberry Pi Connect (no SSH).
- **Hailo stopgap**: YOLO26 Hailo official support is April 2026. If the detector's HEF conversion isn't ready, Phase 5 falls back to YOLO11n. Note: the Hailo-10H model zoo is sparser than Hailo-8's, so the fallback may need its own DFC pass rather than a pre-built HEF.
- **Baseline to beat**: BrunoKreiner's 92–93% mAP50 on the same Kaggle disease dataset (YOLOv8-XL instance segmentation, 2023). Our classifier target: ≥ 90% top-1.

## Repo Layout

```
data/              # Downloaded datasets (gitignored, Pi-resident). 7 sources under data/{zenodo,disease,osf-ej5qv,roboflow/*,strawdi}/
models/            # Trained weights + exports (gitignored). detect/ and disease/
scripts/           # Training + conversion scripts (build_manifest.py, train_detect.py, ...)
notebooks/         # Colab training notebooks (train_detect_colab.ipynb)
src/               # Library code: manifest, backends, pipeline, evaluator
reports/           # CSV + training reports (gitignored)
docs/              # Session logs + Cursor bootstrap prompts
.planning/         # GSD workflow artifacts (committed)
AGENTS.md          # Pi-side agent contract (Cursor execution role)
```

## Running

Training happens on Colab (recommended) or Mac MPS. Inference runs on Pi 5.

- **Detection training (Phase 2)**: open `notebooks/train_detect_colab.ipynb` directly in Colab via https://colab.research.google.com/github/AKarode/strawb-analysis/blob/master/notebooks/train_detect_colab.ipynb. Trains yolo26n (production) + yolo26s (ablation upper-bound) and writes `best.pt` to Drive.
- **Local smoke test**: `python scripts/smoke_test_yolo26.py` (requires ultralytics + a Zenodo sample on disk).
- **Pi-side execution**: read `AGENTS.md`. Cursor on the Pi handles execution; Mac side authors only.

## Brand / Repo Context

This project sits under the parent `Pyranthus/products/` monorepo. See root `CLAUDE.md` for umbrella context. `multi-spectral/` folder name is legacy — actual scope is plain RGB.

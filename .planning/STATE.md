# State: Strawberry Vision Pi

**Initialized:** 2026-04-20
**Last Updated:** 2026-04-20 (post-roadmap)

## Project Reference

- **Project:** Strawberry Vision Pi — Edge Inference Pipeline
- **Core Value:** Single-command batch inference tool that processes 500–1000 field images on a Raspberry Pi within the 30-minute budget, emitting a per-image CSV that lets the client directly compare predictions against annotated ground truth.
- **Current Focus:** Phase 1 planning — data prep & scaffolding.
- **Granularity:** coarse
- **Mode:** yolo
- **Model Profile:** quality

## Current Position

- **Phase:** 1 (Data Prep & Scaffolding)
- **Plan:** — (not yet planned)
- **Status:** Roadmap complete; ready for `/gsd-plan-phase 1`.
- **Progress:** 0/5 phases complete

```
[ ] Phase 1  Data Prep & Scaffolding           (next)
[ ] Phase 2  Detection Model
[ ] Phase 3  Disease Classification Model
[ ] Phase 4  Integrated CPU Pipeline + Evaluator
[ ] Phase 5  Hailo Backend Port                (research flag)
```

## Performance Metrics

| Metric | Value |
|--------|-------|
| Phases defined | 5 |
| Requirements mapped | 25 / 25 (100%) |
| Plans complete | 0 |
| Nodes complete | 0 |
| Phases complete | 0 |

## Accumulated Context

### Key Decisions (from PROJECT.md)

- Two-stage pipeline: YOLO26n detector → YOLO26n-cls disease classifier on cropped fruits (datasets don't permit a unified model).
- YOLO26 over YOLO12 (Ultralytics recommends v11 or v26 for production; v12 marked research-only).
- Original AI HAT+ (Hailo-8) over AI HAT+ 2 (LLM/VLM targeted, no CV benefit).
- NCNN FP32 on CPU path; skip INT8 (not viable on Pi 5 + NCNN as of April 2026).
- Interim Hailo backend: YOLO11n; revisit when YOLO26 Hailo official support ships.
- Disease model: classification, not segmentation (dataset is effectively classification in polygon clothing).
- Synthesize `healthy` class from Zenodo non-diseased fruit crops (disease dataset has no null class).
- Skip Roboflow dataset for v1; Zenodo + Kaggle are sufficient.

### Open TODOs

- Confirm YOLO26 Hailo Model Zoo official support status before planning Phase 5 (via `/gsd-research-phase`).
- Validate the two-stage pipeline assumption during Phase 2 (first real detection results).
- Measure disease model real-world transferability during Phase 3 cross-dataset sanity check.

### Blockers

None.

### Research Flags

- **Phase 5 (Hailo Backend Port):** Hailo YOLO26 official support timing is a project-external dependency. Run `/gsd-research-phase 5` before planning Phase 5 to confirm the current Hailo Model Zoo status. If YOLO26 HEF conversion is available, target it directly; otherwise build against YOLO11n per the stopgap plan.

## Session Continuity

### Last Session Summary

2026-04-20 — Initialized project via `/gsd-new-project`. Captured PROJECT.md, REQUIREMENTS.md (25 v1 reqs), and research SUMMARY.md in-session (no parallel research spawn needed — context already complete). Roadmapper built a 5-phase plan aligned with the natural ML pipeline lifecycle: data → detect → classify → CPU integrate → Hailo port. 100% requirement coverage.

### Next Actions

1. `/gsd-plan-phase 1` — decompose Phase 1 (Data Prep & Scaffolding) into executable plans.
2. Before Phase 5 planning: `/gsd-research-phase 5` to re-verify Hailo YOLO26 support state.

---
*State initialized: 2026-04-20*

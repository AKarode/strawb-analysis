# State: Strawberry Vision Pi

**Initialized:** 2026-04-20
**Last Updated:** 2026-05-09

## Project Reference

- **Project:** Strawberry Vision Pi — Edge Inference Pipeline
- **Core Value:** Single-command batch inference tool that processes 500–1000 field images on a Raspberry Pi within the 30-minute budget, emitting a per-image CSV that lets the client directly compare predictions against annotated ground truth.
- **Current Focus:** Phase 2 functional validation — yolo26n trained on Zenodo, NCNN-exported, awaiting Pi-side benchmark.
- **Granularity:** coarse
- **Mode:** yolo
- **Model Profile:** quality

## Current Position

- **Phase:** 2 (Detection Model) — training done, awaiting Pi benchmark.
- **Plan:** Phase 1 plans 01-01/02 complete; 01-03 (disease crops) + dedup audit still open but unblocking for Phase 2 completion. Phase 2 trained `yolo26n` on Zenodo via Colab (notebook `notebooks/train_detect_colab.ipynb`); NCNN export landed Mac-side; weights + bundle published as GitHub Release `v0.2.0-detect`.
- **Status:** awaiting Pi-side benchmark of `models/detect/yolo26n_zenodo_ncnn_model/` against 67.69 ms reference baseline.
- **Progress:** 0/5 phases complete (Phase 1 ~70%, Phase 2 ~80%)

```
[~] Phase 1  Data Prep & Scaffolding           (~70% — dedup + disease crops outstanding, not blocking P2)
[~] Phase 2  Detection Model                   (training + NCNN export done; Pi bench pending)
[ ] Phase 3  Disease Classification Model
[ ] Phase 4  Integrated CPU Pipeline + Evaluator
[ ] Phase 5  Hailo Backend Port                (research flag — see note)
```

## Phase 2 training summary (2026-05-09)

Trained on Colab A100, dataset = `data/zenodo/strawberries` (654 train / 159 val), 3 classes (ripe / unripe / peduncle).

| Run | Epochs (es) | Best ep | mAP50 | mAP50-95 | precision | recall |
|---|---|---|---|---|---|---|
| yolo26n (production) | 70 | 49 | 0.6791 | 0.3909 | 0.7457 | 0.6585 |
| yolo26s (ablation)   | 67 | 46 | 0.6928 | 0.3887 | 0.7349 | 0.6878 |

Accuracy cost of `n` over `s`: **−0.0022 mAP50-95** (essentially zero — `n` is the right call for the Pi-fit constraint, with no real loss vs. the upper-bound). Note the `s` weights weren't downloaded from Colab (Colab's second `files.download` call dropped); the production `n` is what ships.

Released artifacts (GitHub Release `v0.2.0-detect`):
- `yolo26n_zenodo.pt` (5.1 MB)
- `yolo26n_zenodo_ncnn_model.tar.gz` (8.1 MB compressed, FP32 @ imgsz 640)

## Manifest snapshot (2026-05-05)

Total: 20,313 entries.

| Source | Entries | Notes |
|---|---|---|
| roboflow_afzaal_bbox_v4 | 4,898 | bbox re-annotation of Kaggle Afzaal; includes Roboflow augmentations |
| kaggle_afzaal | 3,243 | LabelMe polygons, 7 disease classes + severity_level_1/2 |
| strawdi | 3,100 | instance masks; **non-commercial — must NOT ship in production weights** |
| osf_ej5qv | 2,967 | classification subdirs; flagged DO NOT TRAIN (provenance unverified, Kaggle overlap suspected) |
| roboflow_research_proj | 2,757 | includes native `healthy_fruit/leaf/flower` classes |
| roboflow_matt_lucky | 2,535 | ripe/unripe bboxes |
| zenodo | 813 | ripe/unripe/peduncle bboxes — primary detection-training source |

## Performance Metrics

| Metric | Value |
|--------|-------|
| Phases defined | 5 |
| Requirements mapped | 25 / 25 (100%) |
| Plans complete | 2 (01-01 LICENSEs, 01-02 manifest library) |
| Models trained | 1 (yolo26n on Zenodo: mAP50 0.679, mAP50-95 0.391) |
| Nodes complete | 0 |
| Phases complete | 0 |

## Accumulated Context

### Key Decisions (current — see PROJECT.md for historical record)

- **Two-stage pipeline**: YOLO26n detector → YOLO26n-cls disease classifier on cropped fruits. Datasets cannot be merged into a unified seg model (no image overlap, incompatible label schemas).
- **YOLO26 over YOLO12** — Ultralytics flags v12 as research-only (training instability). YOLO26 shipped 2026-01-14.
- **Hardware: Raspberry Pi 5 + AI HAT+ 2 (Hailo-10H, 40 TOPS)**. Confirmed by on-device inspection in April 2026; supersedes the original Hailo-8 decision in PROJECT.md / REQUIREMENTS.md (preserved there as historical record).
- **NCNN FP32 only on CPU path** — INT8 on NCNN not viable on Pi 5 as of April 2026.
- **Hailo backend stopgap**: YOLO11n if YOLO26 HEF conversion isn't ready by Phase 5. Hailo-10H model zoo has pre-built HEFs for yolov8n/m/s, yolo11n/x, yolo6n at `/usr/local/hailo/resources/models/hailo10h/` — fallback is essentially trivial.
- **Disease class set**: 7 Kaggle disease classes + native `healthy_fruit` from Roboflow research-proj-disease (originally planned to synthesize healthy crops from Zenodo non-diseased fruits — superseded; native source is cleaner).
- **All 7 datasets in scope** for v1 manifest: zenodo, kaggle_afzaal, osf_ej5qv, roboflow ×3, strawdi. The earlier "skip Roboflow for v1" decision was reversed during dataset inventory — the Roboflow sources fill real gaps (matt-lucky for additional ripeness data, research-proj for native healthy class, afzaal-bbox-v4 for bbox training labels matching the polygon-only Kaggle source).
- **StrawDI is non-commercial-academic**. Manifest includes it for completeness/ablation; trained weights that ship in the commercial deliverable must NOT include StrawDI in their training data. Enforcement is downstream (training code), not at the manifest layer.

### Open TODOs

- **Phase 1**: cross-source dedup audit (`scripts/dedup_audit.py`) — needs authoring; produces `reports/dedup-report.md` per phase 01 CONTEXT.md.
- **Phase 1**: disease crops generation (plan 01-03) — `scripts/build_disease_crops.py` to produce `data/disease_crops/{train,val,test}/<class>/` from Kaggle LabelMe polygons.
- **Phase 5 research flag**: confirm YOLO26 Hailo Model Zoo official support status before planning Phase 5. Hailo's YOLO26 official support landed in April 2026 — verify HEF conversion path for YOLO26n specifically.

### Blockers

None.

### Research Flags

- **Phase 5 (Hailo Backend Port)**: Inverted from highest-risk to lowest-risk after April 2026 on-device validation. Hailo runtime is installed and identifies HAILO10H correctly; pre-built HEFs for several YOLO families are vendor-installed. Phase 5 is now bottlenecked on whether YOLO26n converts cleanly through the Hailo-10H DFC; if not, drop to YOLO11n (pre-built HEF available).

## Session Continuity

### Last Session Summary

2026-05-09 — Phase 2 detection trained + exported. Colab A100 finished both runs (`yolo26n` 70 ep, `yolo26s` 67 ep, both early-stopped); `n` essentially matches `s` on mAP50-95 (Δ −0.0022), so Pi-fit costs nothing. Mac side: created `.venv-export/` (clean torch+ultralytics, anaconda torch was broken), authored `scripts/export_detect_ncnn.py`, produced `models/detect/yolo26n_zenodo_ncnn_model/`. Published GitHub Release `v0.2.0-detect` with `.pt` + NCNN tarball. Authored `scripts/bench_detect_ncnn.py` (Pi-side benchmark + sample-detection writer) and `docs/cursor-phase2-bench.md` (Cursor megaprompt to close Phase 2 in one paste).

2026-05-05 — Public-repo readiness + Phase 1 dataset acquisition. Repo flipped public on GitHub (MIT-licensed, agri-project-goals.docx scrubbed from history). `AGENTS.md` authored to scope the Pi-side Cursor agent's role (execute, don't author). All 7 datasets downloaded on the Pi via Cursor: zenodo direct, OSF via osfclient + inner-zip extract, StrawDI via gdown (Drive interstitial), Kaggle Afzaal via kaggle CLI (`KAGGLE_API_TOKEN` env var), 3× Roboflow via Roboflow Python SDK. Manifest builds end-to-end at 20,313 entries.

2026-04-20 — Initialized via `/gsd-new-project`. Captured PROJECT.md, REQUIREMENTS.md (25 v1 reqs), research SUMMARY.md. Roadmapper built 5-phase plan with 100% requirement coverage.

### Next Actions

1. **Pi-side**: paste `docs/cursor-phase2-bench.md` into Cursor on the Pi → run benchmark → paste `reports/detect_ncnn_bench.md` back. Pass criterion: mean ms/image within ±15% of 67.69 ms baseline.
2. If bench passes, close Phase 2 and start Phase 3 (disease classifier on Kaggle Afzaal crops + native `healthy_fruit` from Roboflow research-proj — same Colab pattern, swap `yolo26n` → `yolo26n-cls`).
3. Phase 1 backfill (not blocking): `scripts/dedup_audit.py` and `scripts/build_disease_crops.py`. Disease crops are needed *before* Phase 3 training, so they slot in between bench-pass and Phase 3 plan.

---
*State initialized: 2026-04-20. Last full refresh: 2026-05-05.*

# State: Strawberry Vision Pi

**Initialized:** 2026-04-20
**Last Updated:** 2026-05-11

## Project Reference

- **Project:** Strawberry Vision Pi — Edge Inference Pipeline
- **Core Value:** Single-command batch inference tool that processes 500–1000 field images on a Raspberry Pi within the 30-minute budget, emitting a per-image CSV that lets the client directly compare predictions against annotated ground truth.
- **Current Focus:** Phase 3 → Phase 4 handoff — disease classifier trained, test-eval verified, NCNN-exported, awaiting release packaging + Pi-side bench.
- **Granularity:** coarse
- **Mode:** yolo
- **Model Profile:** quality

## Current Position

- **Phase:** 3 (Disease Classifier) — **trained + test-eval verified + NCNN-exported.** Ready for release packaging and Pi-side bench.
- **Plan:** Phase 1 plans 01-01/02 complete; 01-03 (disease crops) + dedup audit still open. Phase 2 trained `yolo26n` on Zenodo via Colab; NCNN export Mac-side; release `v0.2.0-detect`; Pi-side bench in `reports/detect_ncnn_bench.md`. Phase 3 trained `yolo26n-cls` on Colab A100 (8-class crops: Kaggle Afzaal × 7 disease + Roboflow research-proj × 1 healthy), test-set eval re-verified Mac-side via `scripts/eval_disease_test.py`, NCNN bundle produced and parity-checked vs PyTorch (identical top-1).
- **Status:** Phase 3 done pending release. Classifier achieves 98.9% val top-1 / **93.86% test top-1 (7-class, healthy not in this eval split)** vs ≥90% target. NCNN export FP32 @ 224×224 produced at `models/disease/yolo26n_cls_ncnn_model/` (5.9 MB bin); NCNN inference matches PyTorch top-1 exactly on the 1416-crop test set.
- **Progress:** 0/5 phases formally complete (Phase 1 ~70%, Phase 2 ~95%, Phase 3 ~90% — pending release + Pi bench)

```
[~] Phase 1  Data Prep & Scaffolding           (~70% — dedup audit outstanding; build_disease_crops landed)
[✓] Phase 2  Detection Model                   (trained, exported, Pi-validated; baseline-revised)
[~] Phase 3  Disease Classification Model      (~90% — trained, test-verified, NCNN-exported; awaiting release + Pi bench)
[~] Phase 4  Integrated CPU Pipeline + Evaluator (scaffolded; runnable once cls NCNN lands on Pi)
[ ] Phase 5  Hailo Backend Port                (research flag — see note)
```

## Phase 2 Pi benchmark summary (2026-05-09)

`yolo benchmark` (canonical Ultralytics method) on Pi 5, 4-thread NCNN, all FP16 + winograd + packing flags optimal:

| Metric | Value | vs published 67.69 ms |
|---|---|---|
| Inference (`yolo benchmark`) | **183.73 ms/image** | 2.71× |
| Inference (`Results.speed`, mean / min) | 197 ms / 126 ms | matches |
| Wall-clock predict pipeline | 388 ms mean | includes ~190 ms JPEG decode of 8 MB source images |

System notes during bench: 4 cores at 2.4 GHz, governor `ondemand` ramped to max under load, `0x80000` throttle bit was historical (sticky), no active throttling, temp 67–72 °C. Cursor agent on same box consumed ~25% CPU.

**Conclusion**: 67.69 ms baseline is unreproducible in our environment (likely ultralytics version delta, vendor benchmark conditions, or the Cursor co-tenant). Empirical 184 ms is the real number. 30-min batch budget still holds with margin.

## Phase 3 disease classifier — training + test eval (2026-05-11)

Trained on Colab A100, 8-class crops (kaggle test crops only excludes `healthy`).

| Run | Epochs (es) | Best ep | val top-1 | val top-5 | params |
|---|---|---|---|---|---|
| yolo26n-cls (production, 50ep config) | 51 | 37 | **0.9898** | 1.0000 | 1.54M |
| yolo26n-cls (rerun, 100ep config)     | 60 | 45 | **0.989**  | 1.0000 | 1.54M |
| yolo26s-cls (ablation) | — | — | (not run) | — | — |

Notebook output drift: the rerun wrote to `disease_yolo26n_cls-2/` in Drive because the original dir existed. Cell 22 / cell 26 read the un-suffixed path, so the downloaded `best.pt` corresponds to the **first 50-ep run**. Both runs converged to ≈98.9% top-1 — the difference is noise; downloaded weights are fine.

### Mac-side test eval (Kaggle test split, 1416 crops, 7 disease classes — `healthy` not in this eval split)

`scripts/eval_disease_test.py --weights models/disease/yolo26n_cls.pt --split test`

| | Overall top-1 | Overall top-5 | Throughput |
|---|---|---|---|
| PyTorch FP32 (MPS) | **0.9386** | 0.9993 | 103 img/s |
| NCNN FP32 (CPU) | **0.9386** | 0.9986 | 184 img/s (Mac) |

Per-class top-1 on test:

| Class | n | top-1 | dominant leak |
|---|---|---|---|
| blossom_blight | 70 | 1.0000 | — |
| gray_mold | 158 | 1.0000 | — |
| powdery_mildew_leaf | 343 | 0.9913 | — |
| leaf_spot | 478 | 0.9226 | → angular_leafspot (2.7%) |
| anthracnose_fruit_rot | 58 | 0.8966 | → powdery_mildew_fruit (6.9%) |
| powdery_mildew_fruit | 116 | 0.8879 | → gray_mold (8.6%) |
| angular_leafspot | 193 | 0.8549 | → gray_mold (5.7%) |

Val→test gap of ~5 pts reveals real failure modes that the val confusion matrix hid: angular_leafspot vs gray_mold, leaf_spot vs angular_leafspot, fungal-fruit cross-confusion. Overall still above the ≥90% target and within range of the BrunoKreiner 92–93% mAP50 baseline (different metric — segmentation vs cls — so not a 1:1 comparison).

NCNN parity: identical top-1 to PyTorch; top-5 drifts by one borderline sample (0.9986 vs 0.9993). Export is clean.

Artifacts:
- `models/disease/yolo26n_cls.pt` (3.2 MB)
- `models/disease/yolo26n_cls_ncnn_model/` (5.9 MB bin + param + metadata.yaml, FP32 @ 224)
- `reports/disease_test_eval.{csv,json}` (PT)
- `reports/disease_test_eval_ncnn.{csv,json}` (NCNN parity)

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

2026-05-11 — Phase 3 closed Mac-side. Colab training completed (see notebook output in `~/Downloads/train_disease_colab.ipynb`); `best.pt` placed at `models/disease/yolo26n_cls.pt`. Authored `scripts/eval_disease_test.py` (per-image CSV + summary JSON, force task=classify so NCNN bundles load as classifiers not detectors). Ran test eval on 1416 Kaggle test crops (7 classes — healthy excluded since Roboflow key wasn't on Mac): **93.86% top-1, 99.93% top-5**. NCNN export via existing `scripts/export_cls_ncnn.py`; re-eval on NCNN bundle confirmed identical top-1 (PT vs NCNN parity check). Reports landed at `reports/disease_test_eval{,_ncnn}.{csv,json}`. Next: package release `v0.3.0-disease` and dispatch Pi-side bench via Cursor megaprompt.

2026-05-09 (evening) — Phase 3 + 4 scaffolded. Authored: `scripts/build_disease_crops.py` (8-class crop generator from Kaggle Afzaal LabelMe polygons + Roboflow research-proj-disease `Healthy Fruit` bboxes), `scripts/train_disease.py` (yolo26n-cls trainer mirroring train_detect.py), `notebooks/train_disease_colab.ipynb` (clone of detect notebook with Kaggle+Roboflow auth + crop generation + training), `scripts/export_cls_ncnn.py` and `scripts/bench_cls_ncnn.py` (cls counterparts to the detect scripts; cls bench computes per-class top-1 accuracy in addition to latency), `src/run_inference.py` (Phase 4 CPU pipeline: detector → per-fruit crop → classifier → CSV with per-image counts and disease set), `src/evaluator.py` (Phase 4 count-accuracy MAE/exact-match/pearson vs Zenodo ground truth), `docs/cursor-phase3-cls-bench.md` (Pi-side megaprompt). Release tagging for Phase 3 mirrors Phase 2: weights + val crops bundle to `v0.3.0-disease`.

2026-05-09 (afternoon) — Phase 2 functionally validated on Pi. Pulled `bench_detect_ncnn.py` from PR #1 merge, ran end-to-end. Initial result was 4.7× the published 67.69 ms baseline; investigation ruled out governor (cores already at 2.4 GHz under ondemand), thread count (NCNN already on 4 with all FP16 + winograd + packing on), and pre/post overhead (only ~12 ms each via `Results.speed`). Wall-clock gap was JPEG decode of 8 MB / 4000×3000 source frames (~190 ms). `yolo benchmark` (the same methodology behind 67.69) on this Pi reports 183.73 ms — confirming the published number is unreproducible in our environment. Updated PROJECT.md and STATE.md with empirical baseline. 30-min/1000-image batch budget holds with margin (~6.5 min detector + ~4 min classifier projected).

2026-05-09 (morning) — Phase 2 detection trained + exported. Colab A100 finished both runs (`yolo26n` 70 ep, `yolo26s` 67 ep, both early-stopped); `n` essentially matches `s` on mAP50-95 (Δ −0.0022), so Pi-fit costs nothing. Mac side: created `.venv-export/` (clean torch+ultralytics, anaconda torch was broken), authored `scripts/export_detect_ncnn.py`, produced `models/detect/yolo26n_zenodo_ncnn_model/`. Published GitHub Release `v0.2.0-detect` with `.pt` + NCNN tarball. Authored `scripts/bench_detect_ncnn.py` (Pi-side benchmark + sample-detection writer) and `docs/cursor-phase2-bench.md` (Cursor megaprompt to close Phase 2 in one paste).

2026-05-05 — Public-repo readiness + Phase 1 dataset acquisition. Repo flipped public on GitHub (MIT-licensed, agri-project-goals.docx scrubbed from history). `AGENTS.md` authored to scope the Pi-side Cursor agent's role (execute, don't author). All 7 datasets downloaded on the Pi via Cursor: zenodo direct, OSF via osfclient + inner-zip extract, StrawDI via gdown (Drive interstitial), Kaggle Afzaal via kaggle CLI (`KAGGLE_API_TOKEN` env var), 3× Roboflow via Roboflow Python SDK. Manifest builds end-to-end at 20,313 entries.

2026-04-20 — Initialized via `/gsd-new-project`. Captured PROJECT.md, REQUIREMENTS.md (25 v1 reqs), research SUMMARY.md. Roadmapper built 5-phase plan with 100% requirement coverage.

### Next Actions

1. **Package release `v0.3.0-disease`**: bundle `models/disease/yolo26n_cls.pt` + tarball of `models/disease/yolo26n_cls_ncnn_model/` + `data/disease_crops/test.tar.gz` (Pi needs the eval crops to measure top-1 on-device). Publish via `gh release create v0.3.0-disease ...` mirroring Phase 2.
2. **(Optional but recommended) 8-class test eval with `healthy`**: re-run `scripts/eval_disease_test.py` with the Roboflow `healthy` crops included. Requires `ROBOFLOW_API_KEY`; either re-add Roboflow's `research-proj-disease` v1 to `data/roboflow/research-proj-disease/`, run `python scripts/build_disease_crops.py` (without `--skip-roboflow`), then re-run eval. Confirms the val 97% healthy holds on test.
3. **Pi side after release is published**: paste `docs/cursor-phase3-cls-bench.md` to bench the cls model and report.
4. **Phase 4 (integrated pipeline)**: `src/run_inference.py` and `src/evaluator.py` are scaffolded and ready. Once both detector + classifier NCNN bundles are on the Pi, the full pipeline runs as `python -m src.run_inference --backend cpu --images ... --detector ... --classifier ... --out reports/inference_cpu.csv`, then `python -m src.evaluator --predictions ... --ground-truth-images data/zenodo/strawberries/validation --out reports/eval_cpu.json`.
5. **Phase 1 backfill (lower priority)**: `scripts/dedup_audit.py` — emits `reports/dedup-report.md`. Not blocking Phase 3 or 4.
6. Optional revisit on Phase 2 throughput: try ultralytics latest (8.4.60+) for a possibly-better NCNN export, or close the Cursor agent on the Pi during the actual demo run.

---
*State initialized: 2026-04-20. Last full refresh: 2026-05-05.*

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

- **Phase:** 2 (Detection Model) — **functionally validated.** Empirical NCNN throughput captured on Pi 5; ready for Phase 3.
- **Plan:** Phase 1 plans 01-01/02 complete; 01-03 (disease crops) + dedup audit still open. Phase 2 trained `yolo26n` on Zenodo via Colab (notebook `notebooks/train_detect_colab.ipynb`); NCNN export landed Mac-side; weights + bundle published as GitHub Release `v0.2.0-detect`; Pi-side bench captured in `reports/detect_ncnn_bench.md`.
- **Status:** Phase 2 done. Detector loads, runs on Pi, produces 3-class detections. Throughput is 184 ms inference / ~390 ms full pipeline per image — slower than Ultralytics' published 67.69 ms benchmark but still meets the 30-min/1000-image batch budget (~6.5 min for detection alone).
- **Progress:** 0/5 phases formally complete (Phase 1 ~70%, Phase 2 ~95% — open Phase 1 backfill items don't block Phase 3 start)

```
[~] Phase 1  Data Prep & Scaffolding           (~70% — dedup audit outstanding; build_disease_crops landed)
[✓] Phase 2  Detection Model                   (trained, exported, Pi-validated; baseline-revised)
[~] Phase 3  Disease Classification Model      (training notebook + bench scripts ready; awaiting Colab run + release)
[~] Phase 4  Integrated CPU Pipeline + Evaluator (scaffolded; runnable once cls NCNN lands)
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

2026-05-09 (evening) — Phase 3 + 4 scaffolded. Authored: `scripts/build_disease_crops.py` (8-class crop generator from Kaggle Afzaal LabelMe polygons + Roboflow research-proj-disease `Healthy Fruit` bboxes), `scripts/train_disease.py` (yolo26n-cls trainer mirroring train_detect.py), `notebooks/train_disease_colab.ipynb` (clone of detect notebook with Kaggle+Roboflow auth + crop generation + training), `scripts/export_cls_ncnn.py` and `scripts/bench_cls_ncnn.py` (cls counterparts to the detect scripts; cls bench computes per-class top-1 accuracy in addition to latency), `src/run_inference.py` (Phase 4 CPU pipeline: detector → per-fruit crop → classifier → CSV with per-image counts and disease set), `src/evaluator.py` (Phase 4 count-accuracy MAE/exact-match/pearson vs Zenodo ground truth), `docs/cursor-phase3-cls-bench.md` (Pi-side megaprompt). Release tagging for Phase 3 mirrors Phase 2: weights + val crops bundle to `v0.3.0-disease`.

2026-05-09 (afternoon) — Phase 2 functionally validated on Pi. Pulled `bench_detect_ncnn.py` from PR #1 merge, ran end-to-end. Initial result was 4.7× the published 67.69 ms baseline; investigation ruled out governor (cores already at 2.4 GHz under ondemand), thread count (NCNN already on 4 with all FP16 + winograd + packing on), and pre/post overhead (only ~12 ms each via `Results.speed`). Wall-clock gap was JPEG decode of 8 MB / 4000×3000 source frames (~190 ms). `yolo benchmark` (the same methodology behind 67.69) on this Pi reports 183.73 ms — confirming the published number is unreproducible in our environment. Updated PROJECT.md and STATE.md with empirical baseline. 30-min/1000-image batch budget holds with margin (~6.5 min detector + ~4 min classifier projected).

2026-05-09 (morning) — Phase 2 detection trained + exported. Colab A100 finished both runs (`yolo26n` 70 ep, `yolo26s` 67 ep, both early-stopped); `n` essentially matches `s` on mAP50-95 (Δ −0.0022), so Pi-fit costs nothing. Mac side: created `.venv-export/` (clean torch+ultralytics, anaconda torch was broken), authored `scripts/export_detect_ncnn.py`, produced `models/detect/yolo26n_zenodo_ncnn_model/`. Published GitHub Release `v0.2.0-detect` with `.pt` + NCNN tarball. Authored `scripts/bench_detect_ncnn.py` (Pi-side benchmark + sample-detection writer) and `docs/cursor-phase2-bench.md` (Cursor megaprompt to close Phase 2 in one paste).

2026-05-05 — Public-repo readiness + Phase 1 dataset acquisition. Repo flipped public on GitHub (MIT-licensed, agri-project-goals.docx scrubbed from history). `AGENTS.md` authored to scope the Pi-side Cursor agent's role (execute, don't author). All 7 datasets downloaded on the Pi via Cursor: zenodo direct, OSF via osfclient + inner-zip extract, StrawDI via gdown (Drive interstitial), Kaggle Afzaal via kaggle CLI (`KAGGLE_API_TOKEN` env var), 3× Roboflow via Roboflow Python SDK. Manifest builds end-to-end at 20,313 entries.

2026-04-20 — Initialized via `/gsd-new-project`. Captured PROJECT.md, REQUIREMENTS.md (25 v1 reqs), research SUMMARY.md. Roadmapper built 5-phase plan with 100% requirement coverage.

### Next Actions

1. **Phase 3 (disease classifier) — Colab training is now ready to run**. Open `notebooks/train_disease_colab.ipynb` directly in Colab via https://colab.research.google.com/github/AKarode/strawb-analysis/blob/master/notebooks/train_disease_colab.ipynb. The notebook auths Kaggle + Roboflow, downloads the source datasets, runs `scripts/build_disease_crops.py` to produce the 8-class training set, and trains both `yolo26n-cls` (production) and optionally `yolo26s-cls` (ablation). Target: ≥ 90% top-1 on val.
2. **Mac side after training completes**: copy `best.pt` to `models/disease/yolo26n_cls.pt`, run `python scripts/export_cls_ncnn.py --weights models/disease/yolo26n_cls.pt`, publish as GitHub Release `v0.3.0-disease` (also include `data/disease_crops/val.tar.gz` so the Pi can measure top-1 in its bench).
3. **Pi side after release is published**: paste `docs/cursor-phase3-cls-bench.md` to bench the cls model and report.
4. **Phase 4 (integrated pipeline)**: `src/run_inference.py` and `src/evaluator.py` are scaffolded and ready. Once both detector + classifier NCNN bundles are on the Pi, the full pipeline runs as `python -m src.run_inference --backend cpu --images ... --detector ... --classifier ... --out reports/inference_cpu.csv`, then `python -m src.evaluator --predictions ... --ground-truth-images data/zenodo/strawberries/validation --out reports/eval_cpu.json`.
5. **Phase 1 backfill (lower priority)**: `scripts/dedup_audit.py` — emits `reports/dedup-report.md`. Not blocking Phase 3 or 4.
6. Optional revisit on Phase 2 throughput: try ultralytics latest (8.4.60+) for a possibly-better NCNN export, or close the Cursor agent on the Pi during the actual demo run.

---
*State initialized: 2026-04-20. Last full refresh: 2026-05-05.*

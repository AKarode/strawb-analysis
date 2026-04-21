# Requirements: Strawberry Vision Pi

**Defined:** 2026-04-20
**Core Value:** A single-command batch inference tool that processes 500–1000 field images on a Raspberry Pi within the 30-minute budget, with per-image CSV output that lets the client directly compare model predictions against annotated ground truth on the exact datasets they supplied.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Data Prep

- [ ] **DATA-01**: Zenodo detection dataset is organized under `data/zenodo/` with YOLO-format labels preserved, training and validation splits respected.
- [ ] **DATA-02**: Kaggle disease dataset is converted from LabelMe polygons to per-polygon cropped classification dataset under `data/disease_crops/` with train/val/test splits preserved from source.
- [ ] **DATA-03**: A `healthy` class is synthesized for the disease classifier by sampling non-diseased fruit crops from the Zenodo dataset, producing a balanced 8th class.
- [ ] **DATA-04**: Dataset conversion is idempotent and reproducible — same source → same crops, every time (deterministic, fixed seed where sampling is needed).

### Detection Model

- [ ] **DETECT-01**: YOLO26n detection model is trained on the Zenodo dataset with classes `ripe`, `unripe`, `peduncle` to completion.
- [ ] **DETECT-02**: Training produces `.pt` weights plus exported ONNX and NCNN artifacts under `models/detect/`.
- [ ] **DETECT-03**: Validation-set mAP@50 and mAP@50-95 are recorded in a training report.
- [ ] **DETECT-04**: A standalone detection-only inference script runs against a folder of images and emits per-image bounding boxes.

### Disease Classification Model

- [ ] **DISEASE-01**: YOLO26n-cls classifier is trained on the prepared disease crop dataset (7 disease classes + synthesized `healthy` = 8 classes).
- [ ] **DISEASE-02**: Training produces `.pt` weights plus exported ONNX and NCNN artifacts under `models/disease/`.
- [ ] **DISEASE-03**: Test-set per-class accuracy and overall top-1 accuracy are recorded in a training report; the model matches or beats 90% top-1 on the held-out Kaggle test split (benchmark target: BrunoKreiner's 92–93% mAP50).
- [ ] **DISEASE-04**: Cross-dataset sanity check is performed: classifier runs on fruit crops produced from Zenodo images and predicted class distribution is inspected for sanity (e.g., not always predicting a single class).

### Inference Pipeline

- [ ] **PIPE-01**: A single script accepts a directory of images and a sample size N (500 or 1000) and runs the full two-stage pipeline (detect → crop → classify) over exactly N images.
- [ ] **PIPE-02**: The script accepts a `--backend` flag with at least `cpu` (NCNN) as a supported value in Phase 4.
- [ ] **PIPE-03**: Per-image output is written to a CSV (one row per input image) with columns including image identifier, total detection count, ripe count, unripe count, peduncle count, per-fruit disease predictions, stage timings (detection ms, classification ms), and total inference time.
- [ ] **PIPE-04**: A warm-up pass runs before the first timed inference so model-load latency is not counted in per-image benchmarks.
- [ ] **PIPE-05**: At end of run, summary statistics are printed: mean / p50 / p95 inference time, total runtime, number of images processed.
- [ ] **PIPE-06**: Pipeline is reproducible — same inputs + same model weights produce identical CSV output across runs (modulo timing columns).

### Hailo Backend

- [ ] **HAILO-01**: A `HailoBackend` implementation runs the detection model as a Hailo HEF file on the Raspberry Pi 5 + AI HAT+ (Hailo-8, 26 TOPS).
- [ ] **HAILO-02**: For the disease classifier, the same Hailo backend runs the classifier as HEF if supported; otherwise falls back to the NCNN classifier for the classification stage (documented clearly).
- [ ] **HAILO-03**: The same `run_inference.py` script accepts `--backend hailo` and produces CSV output in the same schema as the CPU backend.
- [ ] **HAILO-04**: A side-by-side comparison report documents CPU vs Hailo per-image inference time over a representative sample of at least 100 images.

### Evaluation

- [ ] **EVAL-01**: An offline evaluator script reads the CSV output plus original ground-truth annotations and computes detection mAP@50 and mAP@50-95 on images where both predictions and ground truth exist.
- [ ] **EVAL-02**: The evaluator computes per-class disease accuracy against available ground-truth disease labels on whichever dataset split is annotated.
- [ ] **EVAL-03**: Evaluation output is a single human-readable report file (Markdown or plain text) summarizing both detection and disease metrics.

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Data Sources
- **V2-DATA-01**: Pull and integrate the Roboflow `matt-lucky-f7mch/strawberry-k4gtp` dataset as an additional training source if v1 detection mAP is insufficient.
- **V2-DATA-02**: Add real client capture imagery (drone or handheld rig) for true field-deployment validation, once client provides a sample.

### Hailo Migration
- **V2-HAILO-01**: Migrate Hailo detection backend from YOLO11n (interim stopgap) to YOLO26n once Ultralytics + Hailo official support ships and is stable.
- **V2-HAILO-02**: Quantize disease classifier to Hailo HEF if an official path exists and measure end-to-end Hailo throughput vs CPU.

### Robustness
- **V2-ROBUST-01**: Confidence-threshold-based `unknown` bucket for low-confidence disease predictions so the classifier can abstain rather than force a choice.
- **V2-ROBUST-02**: Handle images with zero detected strawberries gracefully (explicit empty-row with zero counts rather than silent skip).
- **V2-ROBUST-03**: Resume/restart support — if a run is interrupted, skip already-processed images and append to existing CSV.

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Multispectral imaging | Not in client docx; folder name is stale. Scope is RGB. |
| Real-time / streaming inference | Client confirmed batch-after-mapping via WhatsApp — all images stored first. |
| Drone / aerial imagery support | Provided datasets are ground-level; no real client rig data available yet. |
| Leaf disease detection in full field scenes | Disease dataset is 419×419 curated crops, not field-scene suitable; leaf classes only work on cropped regions from a leaf detector we don't have. |
| AI HAT+ 2 (Hailo-10H, 40 TOPS) | Targets LLM/VLM workloads; CV pipeline gains nothing from the higher cost. |
| INT8 quantization on Pi 5 + NCNN | Not viable as of April 2026 per published benchmarks. FP32 is fast enough. |
| Training-time model architecture research | Using stock Ultralytics YOLO26 variants; no custom architecture work. |
| Unified detection + segmentation model | Datasets have no image overlap and incompatible label schemas (bbox vs polygon); partial-label training would regress mAP. |
| Web UI, dashboard, or visualization | CLI-only tool per client spec. |
| Live training pipeline on the Pi | Training happens on development workstation / Colab; Pi is inference-only. |

## Traceability

Will be populated by roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | TBD | Pending |
| DATA-02 | TBD | Pending |
| DATA-03 | TBD | Pending |
| DATA-04 | TBD | Pending |
| DETECT-01 | TBD | Pending |
| DETECT-02 | TBD | Pending |
| DETECT-03 | TBD | Pending |
| DETECT-04 | TBD | Pending |
| DISEASE-01 | TBD | Pending |
| DISEASE-02 | TBD | Pending |
| DISEASE-03 | TBD | Pending |
| DISEASE-04 | TBD | Pending |
| PIPE-01 | TBD | Pending |
| PIPE-02 | TBD | Pending |
| PIPE-03 | TBD | Pending |
| PIPE-04 | TBD | Pending |
| PIPE-05 | TBD | Pending |
| PIPE-06 | TBD | Pending |
| HAILO-01 | TBD | Pending |
| HAILO-02 | TBD | Pending |
| HAILO-03 | TBD | Pending |
| HAILO-04 | TBD | Pending |
| EVAL-01 | TBD | Pending |
| EVAL-02 | TBD | Pending |
| EVAL-03 | TBD | Pending |

**Coverage:**
- v1 requirements: 25 total
- Mapped to phases: 0 (pending roadmap)
- Unmapped: 25 ⚠️ (to be resolved by roadmapper)

---
*Requirements defined: 2026-04-20*
*Last updated: 2026-04-20 after initial definition*

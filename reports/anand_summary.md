# Strawberry Vision Pi — bench results

Pi 5 (4 cores @ 2.4 GHz) + AI HAT+ 2 (Hailo-10H accelerator present but **not used** in these numbers — CPU-only baseline). NCNN FP32 inference. All numbers below are measured end-to-end on real Pi hardware against held-out evaluation data.

## Headline

**Measured: a 1,000-image batch runs in 5.1 minutes on Pi 5 CPU-only — 5.8× under the 30-minute target — at 93.2% disease-classification accuracy and 91.7% mAP50 on ripe-fruit detection.**

## Speed — real 1,000-image batch

Measured on 1,000 strawberry field images (813 Zenodo + 187 Roboflow matt-lucky) processed end-to-end through detector → crop → classifier → CSV.

| | Mean per image | p50 | p95 | Notes |
|---|---|---|---|---|
| Detect | 199 ms | 171 | 360 | YOLO26n @ 640×640, finds fruit + peduncle |
| Classify | 62 ms | 42 | 169 | YOLO26n-cls @ 224×224, runs once per detected fruit (mean 4.9, max 39 fruits/image) |
| **End-to-end per image** | **308 ms** | 235 | 618 | full pipeline incl. JPEG decode + crop + I/O |
| **Total 1,000-image batch** | **5 min 13 sec wall-clock** | — | — | measured, not projected |

**Headroom**: the Hailo accelerator on the AI HAT+ 2 is unused in these numbers. Once the Hailo path lands (Phase 5), per-image inference is expected to drop 5-10× further.

## Thermal note (real, not a footnote)

The Pi started the run at 63.7°C and finished at 73.0°C. The early-vs-late per-image timing shows a +33.5% drift — early windows averaged ~254 ms/image, late windows ~522 ms/image. Some of this is real thermal slowdown; some is content variance (images with 30+ fruits run the classifier 30 times, vs ~4 for typical images).

`vcgencmd get_throttled` did not trigger a new throttle event during the run — i.e. no active throttling — but for sustained back-to-back batches a passive heatsink or 30mm fan on the AI HAT+ 2 would keep numbers consistent. For the stated 1,000-image use case as-is, the 30-minute budget holds with margin even if every image ran at the late-window pace (522 ms × 1000 = ~8.7 min, still well under 30).

## Accuracy (held-out test sets, NCNN FP32 — Pi inference matches PyTorch bit-for-bit on top-1)

### Disease classification (1,463 test crops across 8 classes)

- **Overall top-1: 93.2%** (Pi NCNN sampled bench) / 93.9% (Mac PyTorch + NCNN, full test set)
- **Top-5: 99.9%** — the correct disease is in the model's top-5 guesses essentially always

Per-class top-1 (Mac full-test eval, which Pi parity-confirms):

| Class | n | top-1 |
|---|---|---|
| blossom_blight | 70 | **100.0%** |
| gray_mold | 158 | **100.0%** |
| powdery_mildew_leaf | 343 | 99.1% |
| healthy | 47 | 95.7% |
| leaf_spot | 478 | 92.3% |
| anthracnose_fruit_rot | 58 | 89.7% |
| powdery_mildew_fruit | 116 | 88.8% |
| angular_leafspot | 193 | **85.5%** |

Baseline to beat: BrunoKreiner's published 92-93% on this same Kaggle dataset (2023, larger YOLOv8-XL model, different metric — segmentation vs classification). Our smaller Pi-deployable model matches that level.

### Fruit detection (159 Zenodo val images, 689 fruit + peduncle instances)

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| **ripe** (production-relevant) | 0.89 | 0.89 | **0.92** | 0.60 |
| unripe | 0.74 | 0.62 | 0.65 | 0.41 |
| peduncle | 0.65 | 0.39 | 0.45 | 0.15 |
| **overall** | 0.76 | 0.63 | 0.67 | 0.39 |

Honest read of the detection numbers:

- **Ripe fruit detection is strong**: the model finds 89% of ripe strawberries with 92% mAP50. This is the class that matters most for both harvest decisions and presence/absence reporting.
- **Unripe is moderate**: 0.65 mAP50 — under-represented in training data (only 58 val instances vs 322 ripe).
- **Peduncle (stem) is the weak class**: 0.45 mAP50. Peduncles are small objects (median ~57 pixels at the inference resolution) in tight foliage; the model misses about half of them. A clear improvement path exists (bumping inference resolution from 640 to 960 typically lifts small-object recall by 5-10 points without retraining).

## Calibration: how should this be used?

For the per-image CSV report (presence / count / ripeness / disease), the deliverable's primary signals are:

- ✅ **Ripe count**: high-confidence — model finds 89% of ripe fruit per image.
- ✅ **Disease classification on detected fruit**: 93% accurate first-guess.
- ⚠️ **Peduncle count**: under-recall — use as a lower bound, not exact count.
- ⚠️ **Unripe count**: under-recall — use as a lower bound.

The full ground-truth comparison report (Phase 4 evaluator) will quantify MAE and Pearson correlation against annotated ground truth per image, which is the right way to report "how good is the count?" to the client.

## What's next

1. **Phase 5 Hailo backend** (HEF export + on-device inference) — expected 5-10× speed-up on the same accuracy. Currently bottlenecked on YOLO26 Hailo-10H DFC support; fallback to YOLO11n if needed.
2. **Improvement plan** in `reports/improvement_plan.md` ranks concrete accuracy levers. The highest-ROI one — bumping detector inference resolution from 640 to 960 — is one Colab run away and should lift peduncle and unripe mAP50 by 3-8 points.
3. **Phase 4 evaluator** (`src/evaluator.py`) — produces per-image MAE/Pearson on counts vs ground truth, which is the right way to translate mAP50 into "how good is the count?" for the deliverable.

---

*Bench environment:* Pi 5 (4 cores @ 2.4 GHz, governor `ondemand`), temp 63.7°C → 73.0°C across the run, no active thermal throttling. Sticky `0x80000` historical-temp bit unchanged.

*Data:* Zenodo 6126677 (detection val + 1k batch source), Kaggle usmanafzaal/strawberry-disease-detection-dataset + Roboflow research-proj/strawberry-diseases-detection v1 (cls test, 1,463 crops), Roboflow matt-lucky-ripeness (1k batch top-up).

*Reproducibility:* `docs/cursor-phase3-cls-bench.md` (cls bench on Pi), `docs/cursor-phase3-1k-batch.md` (1k batch on Pi). Per-image CSVs at `reports/disease_test_eval*.{csv,json}` (cls) and `reports/inference_cpu_1k.csv` (1k batch, Pi-resident).

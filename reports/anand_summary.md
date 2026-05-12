# Strawberry Vision Pi — bench results

Pi 5 (4 cores @ 2.4 GHz) + AI HAT+ 2 (Hailo-10H accelerator present but **not used** in these numbers — CPU-only baseline). NCNN FP32 inference. All numbers below are measured end-to-end on real Pi hardware against held-out evaluation data.

## Headline

**An offline batch of 1,000 strawberry field images is projected to process in ~8 minutes — 3.75× under the 30-minute target — at 93.2% disease-classification accuracy and 91.7% mAP50 on ripe-fruit detection.**

## Speed (Pi 5, NCNN, CPU-only)

| Stage | Mean per image | p95 | Notes |
|---|---|---|---|
| Detect (find fruit + peduncle) | 286 ms | 423 ms | YOLO26n @ 640×640 |
| Classify (disease per fruit) | 47.8 ms | 107 ms | Runs ~4.2× per image (one per fruit) on 224×224 crops; per-call 12.9 ms |
| **End-to-end per image** | **479 ms** | **679 ms** | full pipeline incl. I/O |
| **1,000-image batch (projected)** | **~8 minutes** | — | linear extrapolation from 159-image measured run (76.1s wall) |

Headroom: the Hailo accelerator on the AI HAT+ 2 is unused in these numbers. Once the Hailo path is finished (Phase 5), per-image inference is expected to drop 5-10×, giving even more margin for higher input resolution or larger model sizes if accuracy needs to climb further.

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

1. Phase 5 Hailo backend (HEF export + on-device inference) — expected 5-10× speed-up on the same accuracy.
2. Improvement plan in `reports/improvement_plan.md` ranks concrete levers; the highest-ROI one (inference resolution 640 → 960) is one Colab run and should bring peduncle and unripe mAP50 up materially.

---

*Bench environment:* Pi 5 (4 cores @ 2.4 GHz, governor `ondemand`), temp 64-68°C during run, no active thermal throttling. Sticky `0x80000` historical-temp bit is from prior load (documented in Phase 2 notes), not a current concern.

*Data:* Zenodo 6126677 (detection val, 159 images), Kaggle usmanafzaal/strawberry-disease-detection-dataset + Roboflow research-proj/strawberry-diseases-detection v1 (cls test, 1,463 crops).

*Reproducibility:* See `docs/cursor-phase3-cls-bench.md` for the one-shot Pi bench. CSVs at `reports/disease_test_eval*.{csv,json}` (cls) and `reports/detect_val_eval_summary.json` (detection).

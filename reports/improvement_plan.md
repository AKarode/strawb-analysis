# Model improvement plan — Phase 2 detector + Phase 3 classifier

Generated 2026-05-11. Grounded in held-out evaluations:

- **Detector** (yolo26n on Zenodo): `reports/detect_val_eval/` + `reports/detect_val_eval_summary.json` (159 val images)
- **Classifier** (yolo26n-cls): `reports/disease_test_eval_8class.{csv,json}` (1463 test crops, 8 classes)

Each lever is scored:

- **Effort**: S (≤1h Mac/Colab) · M (1 Colab run, hours) · L (new data, days)
- **Lift**: rough expected accuracy delta. Multi-lever combinations are not additive.

---

## Current baseline (held-out)

### Detector

| Class | Instances | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| ripe | 322 | 0.885 | 0.894 | **0.917** | **0.600** |
| unripe | 58 | 0.745 | 0.621 | 0.653 | 0.406 |
| peduncle | 309 | 0.648 | **0.385** | **0.446** | **0.154** |
| **all** | **689** | **0.759** | **0.633** | **0.672** | **0.386** |

Dominant failure modes (from `confusion_matrix_normalized.png`):

- **49% of true peduncles are predicted as background** (i.e. missed entirely).
- **27% of true ripes are missed as background**, **28% of true unripes** missed.
- **60% of predictions labeled "peduncle" are actually background** (hallucinated peduncles).

Object-size analysis (rel. bbox area):

- ripe: p50 1.58% (≈80px side @ 640) → easy, well-resolved.
- unripe: p50 0.68% (≈52px), p10 0.08% (≈18px).
- peduncle: p50 0.79% (≈57px), p10 0.21% (≈30px).

Training set: 654 images, 1962 ripe / 747 unripe / 1859 peduncle instances. Unripe instances are 2.6× under-represented vs ripe; peduncle has parity by count but underperforms — i.e. peduncle's gap is **visual distinctiveness + size**, not class frequency.

### Classifier

| Class | n | top-1 | dominant leak |
|---|---|---|---|
| blossom_blight | 70 | 1.000 | — |
| gray_mold | 158 | 1.000 | — |
| powdery_mildew_leaf | 343 | 0.991 | — |
| healthy | 47 | 0.957 | → leaf_spot (4.3%) |
| leaf_spot | 478 | 0.923 | → angular_leafspot (2.7%), → powdery_mildew_leaf (2.5%) |
| anthracnose_fruit_rot | 58 | 0.897 | → powdery_mildew_fruit (6.9%) |
| powdery_mildew_fruit | 116 | 0.888 | → gray_mold (8.6%) |
| angular_leafspot | 193 | 0.855 | → gray_mold (5.7%), → powdery_mildew_leaf (3.6%) |
| **overall** | **1463** | **0.939** | — |

Note val→test gap: val was 0.989, test is 0.939 (Δ −0.050). The val set hid these failure modes — they're a real generalization gap, not noise.

Class imbalance: leaf_spot/powdery_mildew_leaf each have ~10× more training crops than anthracnose/powdery_mildew_fruit/healthy. The four worst-performing classes are all minority classes (n_train ≤ 332). Strong correlation: under-sampled classes underperform.

---

## Detector improvements (ranked by ROI)

### D1. Raise inference imgsz from 640 → 960 (S, +3–8 mAP50)

The single highest-leverage change. Peduncles have p10 size of ~30px at 640 — barely above the receptive-field floor for a YOLO Nano backbone. Bumping to 960 scales small objects to ~45px and typically lifts small-object recall on COCO-like datasets by 5–10 points without retraining.

Train + val at 960:

```bash
yolo train task=detect model=yolo26n.pt data=zenodo_data.yaml \
  imgsz=960 epochs=120 batch=8 patience=20 device=0
yolo val   task=detect model=runs/detect/.../best.pt imgsz=960 data=...
```

Cost: longer Colab run (~2-3× wall time at imgsz=960 vs 640), but Pi inference at 960 is also ~2.3× slower (the detector becomes ~430 ms/img on Pi vs current 184 ms). Still well within the 30-min/1000-image budget (~7 min for detection alone).

**Trade-off**: if Pi budget is tight, train at 960 and infer at 800 — typically retains most of the small-object gain while halving the inference cost. Confirm with bench before committing.

### D2. Test-time augmentation (S, +1–3 mAP50)

`yolo val ... augment=True` enables Ultralytics' TTA (3 scales × horizontal flip). Free at inference — but **only at the bench step**, since enabling TTA in production would 6× Pi inference time and blow the time budget. So this is a "what's the upper bound?" diagnostic, not a deploy lever.

### D3. Train yolo26s instead of yolo26n (S, +1–3 mAP50)

Phase 2 ablation showed n ≈ s for ripe-detection on this set (Δ −0.0022 mAP50-95). But that ablation was at 640 — for small-object recall, the bigger backbone may matter more. Run yolo26s at imgsz=960; if it's > 4 points higher than yolo26n, the size trade-off becomes interesting (s is ~12 MB NCNN bin vs 6 MB — still Pi-fits, slower inference). **Most important: if D1 fixes peduncle, D3 becomes irrelevant.** Run D1 first.

### D4. Augment training with matt-lucky and afzaal-bbox-v4 (M, +2–5 mAP50 on ripe/unripe)

`data/roboflow/matt-lucky-ripeness` (2535 bbox images, ripe/unripe) and `data/roboflow/afzaal-bbox-v4` (4898 bbox images, contains disease + healthy fruit) add ~7400 more bbox-labeled images. Strict caveats:

- They do **not** label peduncles. Mixing label spaces requires either (a) `single_cls` training that loses peduncle, or (b) training a 2-class ripe/unripe head separately + a peduncle head.
- Cross-source label noise is real — Roboflow re-annotations of Kaggle Afzaal had documented quality issues in prior Phase 1 dedup work.
- Doesn't help peduncle directly.

Best execution: train 2-class ripe/unripe on Zenodo+matt-lucky+afzaal-bbox-v4 (10k+ images) as a separate experiment; verify it lifts ripe to >0.95 mAP50 and unripe past 0.80; keep current 3-class Zenodo-only model for peduncle. The pipeline (`src/run_inference.py`) would call both detectors. **High effort and architectural drift — defer until D1+D2 are exhausted.**

### D5. Pseudo-labeling for peduncle on a fresh image source (L, +3–6 peduncle mAP50)

Zenodo's 654 training images are tiny for a detection task with hidden small objects. Pseudo-labels strategy: use the current model to predict on roboflow-research-proj (which has fruit/leaf labels but no peduncles), keep only high-confidence peduncle boxes (>0.6), filter by hand or by spatial-coincidence-with-ripe-fruit heuristics, retrain. Expensive — only worth it after D1.

### D6. Hyper search (M, +1–2 mAP50)

Run `yolo tune` for 30 iterations on Colab A100. Free lift but plateaus fast. Lower priority than imgsz.

### D7. Re-check the 3 corrupt JPEGs in validation (S, +0 — diagnostic only)

`yolo val` flagged `107.jpg`, `12.jpg`, `45.jpg` as corrupt-JPEG-restored. Not a perf issue (ultralytics restores them), but worth noting that the upstream dataset has integrity issues. If matched files exist in training too, the trained model may have learned from minor corruption artifacts.

**Priority for next iteration**: D1 → D2 (diagnostic) → D6 → D3 → D4 → D5.

---

## Classifier improvements (ranked by ROI)

### C1. Class-balanced sampling or weighted loss (S, +2–5 top-1)

The four worst classes (angular_leafspot, powdery_mildew_fruit, anthracnose_fruit_rot, healthy) are all minorities. leaf_spot has 1365 train crops vs anthracnose's 89 — a 15× imbalance. Cheapest fix: weighted sampling.

`scripts/train_disease.py` currently has no class-weighting flag. Add one of:

- **Inverse-frequency weighted sampling**: `WeightedRandomSampler(weights=1/n_per_class)` in the dataloader. Set `model.train(sampler='balanced')` — ultralytics doesn't expose this directly; needs a small patch to the cls trainer or training via raw torchvision.
- **Class-weighted CE loss**: ultralytics 8.x cls trainer accepts a class-balanced loss via `loss=focal` in `cfg`. Try `train ... cls_pw=...` or pass weights via callbacks.

Easiest path: try `model.train(... data=..., augment=True, label_smoothing=0.1, optimizer='AdamW', lr0=1e-3, ...)` with augmentations bumped on minority classes (offline crop oversampling — physically duplicate minority crops with augment baked in).

### C2. Targeted augmentation for confused pairs (S, +1–3 top-1 on weak classes)

The confusion pairs are biologically meaningful — both members of each pair share visual features:

- angular_leafspot ↔ gray_mold (both: brown-ish lesions)
- powdery_mildew_fruit ↔ gray_mold (both: white/grey fuzz on fruit)
- leaf_spot ↔ angular_leafspot (both: leaf lesions; angular_leafspot is a subset visually)

Targeted color/contrast augmentation (HSV jitter) and stronger texture augmentation (cutout, mixup) would help the model rely on local texture rather than coarse hue. Ultralytics cls trainer supports `hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 mixup=0.2 copy_paste=0.0` — bump these. **Easy single Colab rerun.**

### C3. Image size 224 → 320 (S, +1–3 top-1)

Larger input gives more pixels for texture features that separate the confused pairs. Cost on Pi: classifier latency rises ~2×, but classifier runs *after* the detector on tight crops — typically 5–20 crops per image — so even a 2× hit is small in the pipeline budget. Worth trying. Watch overfitting on the smallest classes (anthracnose @ 89 train).

### C4. Add native disease data from afzaal-bbox-v4 healthy/diseased crops (M, +1–3 top-1, mainly on minority classes)

`roboflow/afzaal-bbox-v4` is a re-annotation of the same Kaggle Afzaal source with bbox labels for the same 7 disease classes. Crops from these bboxes are independent samples (different augmentation lineage from Roboflow's pipeline). Could double the training set for minority classes without changing the source domain. Caveat: same-source data — won't help generalization to truly novel conditions; helps within-distribution.

### C5. Two-stage classifier — coarse "type" first, then fine class (M, +2–4 top-1 on confused pairs)

Many confusions are within visual subgroups (`*_leafspot` / `*_mildew_*` / `*_mold_or_rot`). Train a 3-way coarse classifier (leaf / fruit / generic-fungal), then route to specialized fine classifiers. Architecturally clean fix for the biology — but pipeline complexity goes up. Defer until the simpler levers are exhausted.

### C6. Hard-example mining (M, +1–2 top-1)

Use the test-eval CSV to identify the ~89 misclassified crops, manually inspect for label noise, and either (a) drop confirmed bad labels or (b) feed them as oversampled examples. The CSV at `reports/disease_test_eval_8class.csv` is the input.

**Priority for next iteration**: C2 → C1 → C3 → C6 → C4 → C5.

---

## Combined v2 recipe (recommended)

Single Colab run that should produce a v0.4 release:

1. **Detector v2**: yolo26n, imgsz=960, epochs=150, patience=25, augmentations bumped (mosaic=1.0, mixup=0.1). Train on Zenodo only (D1 + D6 light). Target: peduncle mAP50 > 0.55, overall mAP50 > 0.74.
2. **Classifier v2**: yolo26n-cls, imgsz=320, epochs=100, patience=15, HSV jitter and mixup bumped (C2 + C3). Add offline oversampling for minority classes (C1, lite). Target: minority class top-1 floor > 0.92, overall test top-1 > 0.96.

**Effort**: 2 Colab runs (~3–4 hours total on A100), no new data. **Expected combined lift**: detector +5–8 mAP50, classifier +2–4 top-1.

---

## Out of scope (intentionally)

- Multi-scale inference / sliding window: blows the Pi 30-min budget.
- Synthetic data generation (e.g. diffusion-based fruit synthesis): research project on its own.
- Hailo INT8 quantization: deferred to Phase 5; not an accuracy lever.
- Swap to YOLOv8: Phase decision is locked on YOLO26.

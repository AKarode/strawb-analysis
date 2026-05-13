# Anand call prep — 2026-05-13

Reference doc for the call about the strawberry vision pipeline results. Headline numbers + answers to questions Anand is most likely to ask.

## 30-second headline

- 1,000 strawberry field images processed in **5 min 13 sec** on Pi 5 CPU.
- Target was 30 min — **5.8× under budget**.
- Disease classification: **93% top-1** across 8 classes.
- Ripe-fruit count: **off by 0.28 fruit per image on average**, exactly right 77% of the time.
- Hailo accelerator on the Pi is not even used yet — that's the future 5-10× speed-up.

## Q: Walk me through the dataset sampling

**Three sources, picked for non-overlapping coverage of the deliverable's two tasks (detection + disease classification):**

| Dataset | What it gives us | How many | License |
|---|---|---|---|
| Zenodo 6126677 (Pastell et al.) | Field strawberry photos with bbox annotations: **ripe / unripe / peduncle** | 813 imgs (654 train + 159 val) | CC-BY 4.0 (open) |
| Kaggle Afzaal | Disease photos with LabelMe polygons: **7 disease classes** | 3,200 source imgs → 5,960 cropped fruit/leaf | License "unknown" — used for training, not redistributed |
| Roboflow research-proj-disease | Native **healthy** annotations to balance the disease classes | 250 healthy crops | Roboflow standard terms |

**Sampling decisions:**

- **Train/val/test splits are upstream** — we use the dataset authors' splits. Zenodo gives us 654 train / 159 val (no test); Kaggle gives us 1,450/307/743 train/val/test images. This avoids any accusation of cherry-picking splits.
- **Disease crops come from the polygon/bbox annotations, not full images.** Each strawberry/leaf is cut out as a 224×224 crop with 10% padding around the bounding box, because that's how the classifier sees fruit at inference time (after the detector hands it cropped boxes). The training input distribution matches the production input distribution.
- **Class imbalance is real and acknowledged**: leaf_spot has 1,365 train crops, anthracnose_fruit_rot has 89 (15× imbalance). The four weakest classes in our test eval are all minority classes. Improvement plan ranks weighted sampling as a top fix.
- **What we didn't use and why**: OSF dataset (provenance unverified, suspected overlap with Kaggle); StrawDI (non-commercial-academic license — can't ship trained weights from it). matt-lucky and afzaal-bbox-v4 Roboflow sets are available for v2 expansion but weren't needed for v1 baseline.

## Q: Which models, and why?

**Detector**: Ultralytics **YOLO26n** at 640×640 input.

- YOLO26 is the current Ultralytics flagship, released **January 14, 2026**. YOLOv12 is technically newer numerically but Ultralytics themselves flag v12 as research-only (training instability).
- **n** (nano) size: 2.4M params. We tested **s** (small, 11M params) as an ablation — gave only **+0.0022 mAP50-95**, essentially zero improvement, for ~4× the latency cost. Going small is the right Pi-fit call.

**Classifier**: Ultralytics **YOLO26n-cls** at 224×224.

- Same family, smallest size: 1.5M params, 3.2 GFLOPs, 3 MB file.
- Trained on tight crops (224×224) because that's what the detector hands it at inference time.
- Hit 98.9% val top-1 in training; 93.9% on held-out test. Did not run an s-size ablation since n already crushes the target.

**Why two stages instead of one segmentation model?**

The datasets are mutually exclusive. Zenodo annotates ripe/unripe/peduncle bboxes but doesn't label disease. Kaggle Afzaal annotates disease but doesn't distinguish ripe/unripe. There is no overlapping image source that has both labels on the same fruit. A unified segmentation model would need that — without it, two-stage is the only viable architecture.

## Q: Why not use the Hailo AI HAT+ 2 accelerator?

**It's installed and confirmed working. We just didn't need it for v1.**

- The contract budget is 30 minutes per 1,000 images.
- Our CPU-only pipeline does the job in 5 min 13 sec. That's **5.8× under budget**.
- The AI HAT+ 2 (Hailo-10H, 40 TOPS) would give an additional **5-10× speed-up** — i.e. processing the same 1,000 images in 30-60 seconds.
- We deferred that work (Phase 5) because:
  1. It doesn't unblock the deliverable.
  2. The Hailo Dataflow Compiler path for YOLO26 specifically is a research item — Hailo officially added YOLO26 support in April 2026, so we'd need to do quantization + calibration work that could take 1-3 days to land cleanly, vs the days of certainty we already have shipped.
  3. We have a fallback path (YOLO11n) if YOLO26 conversion proves finicky.

**Frame this as a strategic choice**: "We have 5-10× of headroom available on this hardware — sized for tomorrow's requirements, not just today's."

## Q: How long would this take on a drone with live video?

**Short answer**: technically feasible to do onboard at 3 fps on Pi CPU, or 20-30 fps with the Hailo NPU. Most practical product shape is "drone records flight, Pi processes after landing." Here's the math:

### Per-frame numbers (already measured)

- Pi 5 CPU pipeline: **308 ms per frame** (199 ms detect + 47 ms cls × ~4-5 fruits/frame).
- Pi sustained throughput: **~3.2 frames per second**.
- With the Hailo NPU enabled (Phase 5): estimated **30-50 ms/frame, 20-30 fps**.

### Drone flight math

Assume a typical agricultural drone (DJI Mavic 3 / Phantom-class):

| Parameter | Typical value |
|---|---|
| Altitude | 10-20 m |
| Speed | 3-7 m/s |
| Camera FOV | 70-80° diagonal |
| Ground footprint at 10 m alt | ~12 m × 8 m |
| Useful overlap (for mapping) | 30-50% |

### Coverage example: 1 hectare (10,000 m² ≈ 2.5 acres)

- Effective area per frame with 50% overlap: ~48 m²
- Frames needed for full coverage: **~210 frames**
- Drone flight time at 5 m/s lawnmower pattern: **~7 minutes**
- Frame capture rate to maintain overlap: **~1 fps** (one frame every 1.1 seconds)

### Processing scenarios

| Architecture | Pi load | When you get results | Notes |
|---|---|---|---|
| **A. Onboard real-time (CPU only)** | 1 fps capture → 3.2 fps capacity = 3× headroom | Live, during flight | Power: Pi 5 + camera ≈ 10 W draws ~1 Ah/hr from drone battery — may need supplementary power. Heat is a real factor for sustained operation. |
| **B. Onboard recorded, offline batch** (Recommended for v1) | Drone records video + GPS to SD card; Pi processes after landing | ~3 min processing per 7-min flight | Same pipeline we already have, no architectural changes. Clean separation between flight ops and CV ops. |
| **C. Onboard real-time with Hailo** (Phase 5) | 20-30 fps capacity — could capture & process 4K video frame-by-frame | Live, with margin to display annotations | Needs Phase 5 Hailo port. Real selling point for higher-end product tier. |

### Geo-locating each strawberry

Each frame gets a GPS coordinate from the drone's flight log (typically 10-20 Hz GPS, time-synced to frame timestamps). Two pieces are needed:

1. **Frame timestamp → GPS coordinate**: interpolate the drone's flight log to each frame's capture time. Provides the *camera position* in WGS84 lat/lon + altitude.
2. **Pixel position → ground position**: given camera intrinsics (focal length, sensor size) + drone pose (roll/pitch/yaw) + altitude, project each detection's bounding-box center to a ground coordinate. This is standard photogrammetry — well-supported by libraries like OpenCV / OpenDroneMap.

Output is a list: `(lat, lon, ripe_count, disease_present)` per detection, or a heatmap aggregated to a grid.

### Deduplication across overlapping frames (a real product concern)

If a strawberry appears in 3 overlapping frames (because of 50% overlap), naive aggregation triple-counts it. Real-world solutions:

- **Spatial clustering**: cluster detections by ground coordinate; nearby points within ~5 cm = same berry.
- **Visual odometry / SLAM**: track frames spatially so the same fruit is recognized across frames.
- **Stride frames**: skip every other frame to eliminate overlap (loses some count accuracy but simpler).

This is engineering work, ~1-2 weeks. Not in current scope but well-understood.

### Realistic timeline if Anand wants the drone integration

| Milestone | Effort | Output |
|---|---|---|
| Add video → frames step to pipeline | 0.5 day | `ffmpeg`-based frame extractor, integrates cleanly with `src/run_inference.py` |
| GPS sync from drone telemetry | 1-2 days | Per-frame lat/lon/altitude |
| Pixel-to-ground projection | 2-3 days | OpenCV photogrammetry, validated against ground-truth markers |
| Detection deduplication | 3-5 days | Cluster-based dedup with tuning |
| Field heatmap output | 1-2 days | GeoJSON / KML output for QGIS / Google Earth |
| Hailo port for onboard real-time (optional) | 1-2 weeks | Phase 5 — 5-10× speed up |
| **Total to MVP drone integration** | **~2-3 weeks** | End-to-end: drone flies → field heatmap |

This is on top of the current deliverable — the current pipeline can be the per-frame processor inside this larger system without modification.

## Q: What's the accuracy story honestly?

Anand might press on weaknesses. Honest answers:

| Class | Status | Read |
|---|---|---|
| **Ripe fruit detection** | mAP50 **0.917**, count MAE 0.28/img, 77% exact-match | Strong. The number that matters for harvest decisions. |
| **Disease classification (overall)** | top-1 **0.939**, top-5 **0.999** | Strong. Above the BrunoKreiner 92-93% baseline (the public reference). |
| Unripe fruit detection | mAP50 0.65, count MAE 0.24/img, 81% exact-match | Moderate — fewer training examples than ripe. |
| **Peduncle (stem) detection** | mAP50 **0.45** | **The weak class.** Small objects, missed about half. Known fix: bump input resolution from 640×640 to 960×960 (one Colab run, +3-8 mAP50 expected). |
| Disease class: angular_leafspot | 0.855 top-1 | Weakest disease class; confused with gray_mold. Visually similar lesions. Fixable with class-balanced sampling + targeted augmentation. |

**The honest punchline**: "The ripe count is essentially exact in 77% of images and off by less than 1 fruit when it isn't. The stem count is a lower bound — we know exactly why (small object at our current input resolution), and we know the fix (re-train at higher resolution, one day of work)."

## Q: What's still on the table?

1. **Phase 1 dedup audit** — non-blocking, just cleanup of the training data.
2. **Detector v2 at 960×960** — top item on the improvement plan. Lifts peduncle and unripe mAP50 by 5-8 points. One Colab run.
3. **Classifier v2 with class balancing** — closes the minority-class gap (angular_leafspot, powdery_mildew_fruit, anthracnose_fruit_rot). One Colab run.
4. **Phase 5 Hailo backend** — when latency requirements tighten or batch sizes grow significantly.
5. **Drone integration** — see above, ~2-3 weeks for MVP.

## Q: What's it cost to deploy in the field?

- Raspberry Pi 5 (8 GB): **$80**
- AI HAT+ 2 (Hailo-10H, 40 TOPS): **$120**
- Storage (256 GB SD or NVMe): **$25**
- Camera (USB or Pi camera): **$30-100**
- **Total per node: $250-350**

vs cloud GPU inference: ~$0.50-2/hour, but requires connectivity (no offline ops).

## Q: Do I need to label my photos before processing?

**No.** Critical clarification — Anand may ask this and the answer is fundamental:

- **Labels (bboxes, polygons) are only needed for *training*.** That step is done. We used public datasets (Zenodo, Kaggle Afzaal, Roboflow) that came pre-labeled, and trained the models on those.
- **At inference (production), photos come in raw — no labels, no annotations.** The detector generates its own bboxes from scratch: "I see a ripe strawberry here, at this rectangle, with 92% confidence." The classifier then predicts the disease for each detected fruit. Output is the per-image CSV.
- The Phase 4 evaluator (`src/evaluator.py`) does use labels, but only for **measuring accuracy** against 159 Zenodo val images. Once Anand deploys, no comparison happens — just predictions go out.

**One caveat to mention if pressed**: if his field photos look drastically different from our training data (different lighting, camera angle, growth stage, strawberry variety), accuracy could drop. The fix is a small **fine-tuning pass**: label ~100-200 photos from his actual fields, retrain for a day, deploy the adjusted weights. Standard practice. Not needed unless we observe a real drop on his data.

The two-stage pipeline in plain English:
1. Photo of strawberry field → detector finds every fruit + draws a rectangle around each → "5 strawberries here."
2. Each rectangle is cropped → classifier looks at just that fruit → "this one is healthy, this one has gray_mold."
3. Write all of that to one row of the output CSV.

## Q: Can we use non-RGB imagery (multispectral, thermal, etc.)?

**Short answer**: yes, but it's a real scope expansion — not a switch-flip. There's a fork to surface for Anand.

### What "non-RGB" typically means in ag

| Type | What it captures | Common ag use | Camera cost |
|---|---|---|---|
| Multispectral (R, G, B, NIR, Red-edge) | Visible + near-IR + stress signal | NDVI / NDRE for plant stress, water status, early disease | $3.5k-10k (MicaSense, Parrot Sequoia) |
| Thermal / LWIR | 8-14 μm heat radiation | Water stress, canopy temperature, irrigation tuning | $1.5k-3k (FLIR Vue, Workswell) |
| Hyperspectral (100+ narrow bands) | Fine-grained spectral signatures | Sugar content, anthocyanin, specific compounds | $20k+ (research-grade) |
| NIR-only single band | Near-infrared 700-1000nm | Cheap proxy for NDVI | $200-500 |

For a winery client specifically, **multispectral for NDVI / canopy stress** is the most common ask.

### What works as-is (cheap add)

**NDVI and similar vegetation indices don't need ML.** They're math on spectral bands:

```
NDVI = (NIR - Red) / (NIR + Red)
```

If the request is "give me NDVI heatmaps," that's a **1-week add** to the pipeline: read multispectral input, compute indices per pixel, emit a heatmap. No new training, no new datasets.

### What needs real work (ML on multispectral)

| Item | Effort |
|---|---|
| Modify YOLO architecture to accept N input channels | ~1 week |
| Find or commission multispectral training dataset | **Big variable** — public multispectral strawberry/grape datasets are rare. 2 weeks to find one, or 2 months to commission. |
| Retrain detector + classifier on multispectral | 1-2 weeks |
| Re-export to NCNN for Pi | 1-2 days |
| **Total** | **3-8 weeks**, gated on dataset availability |

The bottleneck is data, not architecture. RGB strawberry/grape data is plentiful; multispectral data is genuinely scarce.

### Three questions to ask Anand on the call

1. **"What kind of non-RGB — multispectral (NDVI), thermal, or hyperspectral?"** Wildly different effort profiles.
2. **"Do you have or are you sourcing the camera, or are we picking it?"** Constrains the rest of the pipeline ($1.5k-20k camera range).
3. **"Is the goal a new signal (NDVI as an additional output) or replacing the RGB pipeline with multispectral inputs?"** Adding a signal is cheap (~1 week); replacing is 3-8 weeks.

### One-line answer for the call

> "The architecture is multispectral-portable — YOLO can take any number of input channels, and the Pi can absolutely process multispectral imagery. The two real gates are camera cost and training data. NDVI-style vegetation indices don't need ML at all and are a 1-week add. ML on multispectral input — detecting diseases earlier via the red-edge band — is 3-8 weeks depending on dataset availability."

### Don't oversell

- Don't promise "multispectral works out of the box." Current weights are RGB-only.
- Don't promise to find a multispectral dataset until you've actually searched — could require commissioning new labels.
- Don't blur the line between "NDVI as a math output" (cheap) and "ML on multispectral input" (expensive). Very different conversations.

## Q: How does this translate to a winery client?

**Headline**: about 80% of the work moves over directly. The 20% that doesn't is the part you'd expect — different visual domain means different training data and different class labels. Realistic translation timeline: **3-5 weeks** to a winery-equivalent deliverable.

### What transfers with zero modification

- **Architecture**: detector → per-instance crop → classifier → per-image CSV → evaluator. This is the standard agricultural CV pattern and works for any "find objects + assess each object" task — grape bunches, apples, lettuce heads, anything.
- **Pi deployment stack**: NCNN export, Pi 5 + AI HAT+ 2 hardware, the whole "offline batch inference at 5+× under budget" story holds. Speed numbers translate directly because they're measured on this hardware, not on this dataset.
- **All tooling**: training scripts (`train_detect.py`, `train_disease.py`), NCNN export scripts, Pi bench scripts, count evaluator, integrated pipeline. Just point at different data + classes.
- **Drone integration approach**: GPS sync, photogrammetry, geo-tagged detections — identical workflow.
- **Hailo 5-10× headroom story**: same hardware, same speed ceiling.
- **PR + audit infrastructure**: manifest builder, per-source LICENSE tracking, dedup audit pattern. Reusable governance.

### What needs new work

- **Training datasets** — strawberry weights don't transfer to grapes (different color, shape, occlusion patterns, cluster structure). Need grape-specific data.
- **Class labels** — wineries care about different things: bunch counts, disease (powdery mildew, downy mildew, botrytis bunch rot, esca, leafroll virus, black rot — overlapping with grape disease taxonomy), maturity / phenology stage (flowering, fruit set, veraison, harvest-ready), pruning indicators.
- **Detection granularity decision** — most grape datasets label *bunches* not individual berries. Whether the winery wants bunch counts or berry-per-bunch counts is a scoping conversation. Bunch counts are way easier; per-berry counts inside a bunch are a harder research problem because of 3D occlusion.

### Available grape datasets (rough recall — verify before quoting)

- **WGISD (Wine Grape Instance Segmentation Dataset)** — Embrapa, ~300 images across 5 grape varieties with instance masks for bunches. CC-BY 4.0. Most directly comparable to our Zenodo source.
- **PlantVillage grape subset** — disease classification, 4 classes: Black rot, Esca (Black Measles), Leaf blight (Isariopsis Leaf Spot), healthy. Strong analog to our Kaggle Afzaal disease source.
- **Roboflow Universe** has multiple vineyard / grape detection sets of varying quality.
- **Embrapa Vineyard datasets** beyond WGISD — sometimes paired with multispectral.
- Many viticulture research papers publish datasets that aren't on the main hubs — would do a focused search before scoping.

### Translation timeline (Phase-by-Phase, same playbook)

| Phase | Strawberry effort | Grape estimate | Notes |
|---|---|---|---|
| 1. Data prep + manifest | ~3 weeks | **1-2 weeks** | Reuse scripts; download new datasets; build manifest. Same governance pattern. |
| 2. Detection (bunch + maybe leaf/flower) | ~1 week | **1 week** | Identical training pipeline; just different class set + dataset. |
| 3. Disease classification | ~2 weeks | **1-2 weeks** | PlantVillage grape data is well-curated; should converge faster than strawberry did. |
| 4. Integrated pipeline + evaluator | ~3 days | **<1 day** | Pipeline code unchanged; swap weights. |
| 5. Hailo backend | deferred | deferred / same | Same DFC story; not blocking. |
| **Total to v1 winery deliverable** | | **3-5 weeks** | Tight scope, single playbook execution. |

### Winery-specific things to flag (or not)

- **Wineries care more about quality than count.** A grape grower wants yield estimation, but a winery downstream is more interested in fruit quality (brix, anthocyanin, disease pressure). Pure CV is one input among several (refractometer readings, lab analysis). Frame the deliverable as "automated visual triage that reduces hours of scouting" rather than "replaces the field manager."
- **Multispectral / NDVI is more common in viticulture than strawberry.** Vineyard PA (precision agriculture) often uses NIR/red-edge bands for canopy stress / water status. If the winery wants that, our "RGB-only" stance becomes a v1 limitation; we'd plan v2 with a multispectral camera. **Worth asking Anand directly: is the camera spec RGB or multispectral?**
- **Phenology tracking is a v2+ ask.** Knowing when veraison happens at a per-vine level is high-value but requires temporal data (same vine, multiple flyovers), not just classification.
- **Variety-specific models.** Cabernet looks different from Chardonnay. Either train per-variety or train a general model with variety conditioning. Most published work goes per-variety. WGISD has 5 varieties as separate splits — directly usable.

### One-line answer if Anand presses

> "The whole stack — training, deployment, drone integration, the Pi + Hailo story — is winery-portable. The strawberry-specific bits are the training data and the class labels. We'd expect 3-5 weeks to land a winery-equivalent v1, using public grape datasets like WGISD and PlantVillage as the analog of what Zenodo and Kaggle gave us here. The biggest scoping question is whether the deliverable counts bunches or assesses berry-level detail — those are different effort levels."

## Things to NOT say on the call

- "Hailo is broken / didn't work" — it works, we just didn't need it.
- "More data would solve it" — not the bottleneck. Architecture + resolution levers are stronger.
- "It's only 90%" — frame as "0.28 MAE per image" (concrete) instead of just an overall percentage.
- Don't oversell real-time onboard processing without acknowledging the power/thermal tradeoffs.

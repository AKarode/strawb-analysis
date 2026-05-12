# Cursor megaprompt — Phase 3 cls bench + Phase 4 integrated pipeline (Anand handoff)

Paste the block below into a fresh Cursor chat on the Pi. It closes the loop in one shot: pulls the latest repo, fetches the cls NCNN bundle from release `v0.3.0-disease`, rebuilds disease crops from Phase 1 raw data, runs the cls latency + accuracy bench, then runs the **integrated CPU pipeline** (`src/run_inference.py`) on the Zenodo validation set to get end-to-end per-image batch timing. Reports back everything we need to send Anand a single accuracy-and-speed summary.

Pre-conditions on the Pi:
- Repo cloned (Phase 1 + 2 bootstraps already done).
- `.venv` already primed from Phase 2 (`ultralytics`, `ncnn`, CPU torch).
- Phase 1 raw datasets present at `data/disease/{train,val,test}/` (Kaggle Afzaal) and `data/roboflow/research-proj-disease/{train,valid,test}/` (Roboflow). Verify with `ls data/disease/ data/roboflow/research-proj-disease/`.
- Phase 2 detector NCNN bundle present at `models/detect/yolo26n_zenodo_ncnn_model/` (Phase 2 release).
- Zenodo validation images at `data/zenodo/strawberries/validation/` (159 .jpg).

---

You are the Pi-side Cursor agent for Strawberry Vision Pi (read `AGENTS.md` — execute-only, no authoring, no commits to master).

Phase 3 disease classifier is published as GitHub Release `v0.3.0-disease`. Your job is to (1) bench the cls model on this Pi 5, (2) verify its accuracy on local disease crops, and (3) run the integrated CPU pipeline end-to-end on Zenodo's 159 val images so we have a real per-image batch time for the client.

Do this in one shot — don't pause for confirmation unless a step actually fails.

## Step 1 — Sync repo + activate venv

```bash
REPO=$(find / -maxdepth 6 -type d -name strawb-analysis 2>/dev/null | head -1)
[ -z "$REPO" ] && { echo "no clone found"; exit 1; }
cd "$REPO"
git pull --ff-only
source .venv/bin/activate
python -c "import ultralytics, ncnn; print('ultralytics', ultralytics.__version__, 'ncnn', ncnn.__version__)"
```

## Step 2 — Pull the cls NCNN bundle from release `v0.3.0-disease`

```bash
mkdir -p models/disease
cd models/disease
if [ ! -d yolo26n_cls_ncnn_model ]; then
  curl -sSL -o ncnn.tar.gz https://github.com/AKarode/strawb-analysis/releases/download/v0.3.0-disease/yolo26n_cls_ncnn_model.tar.gz
  tar -xzf ncnn.tar.gz
  rm ncnn.tar.gz
fi
ls yolo26n_cls_ncnn_model/
grep -E '^(task|imgsz|names):' -A 8 yolo26n_cls_ncnn_model/metadata.yaml
cd "$REPO"
```

## Step 3 — Rebuild disease crops locally (license-clean — no crop redistribution)

```bash
# Phase 1 raw data -> 8-class crops. ~5s on Pi.
python scripts/build_disease_crops.py \
  --kaggle-root data/disease \
  --roboflow-root data/roboflow/research-proj-disease \
  --out data/disease_crops \
  --pad-pct 0.10

# Expected: test split should be 1463 crops across 8 classes
# (193 angular_leafspot, 58 anthracnose_fruit_rot, 70 blossom_blight, 158 gray_mold,
#  47 healthy, 478 leaf_spot, 116 powdery_mildew_fruit, 343 powdery_mildew_leaf).
find data/disease_crops/test -name '*.jpg' | wc -l
ls data/disease_crops/test/
```

## Step 4 — Cls latency + accuracy bench

```bash
echo "=== pre-bench env ==="
vcgencmd measure_temp
vcgencmd get_throttled
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

OMP_NUM_THREADS=4 python scripts/bench_cls_ncnn.py \
  --model "$REPO/models/disease/yolo26n_cls_ncnn_model" \
  --crops "$REPO/data/disease_crops/test" \
  --warmup 10 \
  --iters 500

echo "=== post-bench env ==="
vcgencmd measure_temp
vcgencmd get_throttled
```

Expected bands (preliminary — confirm against measurement):

- **Latency**: 5-25 ms/image at 224x224 (cls is much smaller than detect's 184 ms @ 640).
- **Accuracy**: should be near Mac NCNN's 0.939 top-1. Top-1 dropping below 0.90 -> suspect bundle corruption or label-order mismatch.

## Step 5 — End-to-end integrated pipeline (the Anand number)

This is what matters for the client: how long does it take to process 159 strawberry field images at the stated accuracy?

```bash
mkdir -p reports
time python -m src.run_inference \
  --backend cpu \
  --images data/zenodo/strawberries/validation \
  --detector models/detect/yolo26n_zenodo_ncnn_model \
  --classifier models/disease/yolo26n_cls_ncnn_model \
  --out reports/inference_cpu_e2e.csv

# Aggregate per-image timings from the CSV
python - <<'PY'
import csv, statistics
with open('reports/inference_cpu_e2e.csv') as fh:
    rows = list(csv.DictReader(fh))
det = [float(r['detect_ms']) for r in rows if r.get('detect_ms')]
cls = [float(r['classify_ms']) for r in rows if r.get('classify_ms')]
tot = [float(r['total_ms']) for r in rows if r.get('total_ms')]
n_fruit = [int(r['n_total']) for r in rows if r.get('n_total')]
def stat(x, label):
    if not x:
        print(f"{label}: no data"); return
    x_s = sorted(x)
    print(f"{label:12s} n={len(x):4d}  mean={statistics.fmean(x):8.1f} ms"
          f"  p50={x_s[len(x_s)//2]:8.1f}  p95={x_s[int(len(x_s)*0.95)-1]:8.1f}"
          f"  min={min(x):.1f}  max={max(x):.1f}")
print(f"\nimages processed: {len(rows)}")
stat(det, "detect")
stat(cls, "classify")
stat(tot, "total")
total_s = sum(tot) / 1000
print(f"\nbatch wall time (sum of per-image total): {total_s:.1f}s "
      f"-> projected for 1000 images: {total_s * 1000 / len(rows) / 60:.1f} min")
if n_fruit:
    print(f"fruits per image: mean {statistics.fmean(n_fruit):.1f}, "
          f"p50 {sorted(n_fruit)[len(n_fruit)//2]}, max {max(n_fruit)}")
PY
```

The projection at the end is the headline number we'll send to Anand: "X minutes for 1000 images at Y% accuracy."

## Step 6 — Compose Anand-ready summary

```bash
cat > /tmp/anand_summary.md <<'EOF'
# Strawberry Vision Pi — bench results (Pi 5 + AI HAT+ 2)

(Auto-generated from the Step 4 + Step 5 outputs. Edit before sending.)

## Speed (Pi 5 CPU, NCNN FP32)

- Detector: <det mean> ms/image @ 640x640
- Classifier: <cls mean> ms/image @ 224x224 (runs N=<mean fruits/image> times per image)
- End-to-end per image: <tot mean> ms (p95 <tot p95> ms)
- Projected 1000-image batch: <X> minutes (target was 30 min; budget holds)

## Accuracy (held-out)

- Detection (Zenodo val, 159 images, 689 fruit instances):
  - mAP50 overall: 0.672
  - ripe: 0.917 mAP50 (the production-relevant class)
  - unripe: 0.653 mAP50
  - peduncle: 0.446 mAP50 (small object, lowest)
- Disease classification (1463 held-out test crops, 8 classes):
  - Top-1: 0.939 (PyTorch FP32 = NCNN FP32, bit-identical)
  - Strong classes (>=0.99): blossom_blight, gray_mold, powdery_mildew_leaf
  - Weakest class: angular_leafspot at 0.855 (confused with gray_mold)

## Hardware

- Raspberry Pi 5, 4 cores @ 2.4 GHz, governor ondemand
- AI HAT+ 2 (Hailo-10H, 40 TOPS) -- NOT used for these numbers; CPU-only.
  The Hailo path (Phase 5) is expected to be 5-10x faster but is still in
  research. The CPU numbers above are sufficient for the 30-min/1000-image
  budget.
EOF
cat /tmp/anand_summary.md
```

## Report back

1. The trailing summary block from `scripts/bench_cls_ncnn.py` (latency + per-class accuracy).
2. The `time` output from Step 5 AND the aggregated per-image timings printed by the inline Python.
3. The composed `/tmp/anand_summary.md`.
4. Any warnings or failures.

## Pass criteria

- **Cls latency**: mean within the 5-25 ms band, no thermal throttling during the 500-iter run.
- **Cls accuracy**: top-1 within +/-0.02 of Mac's 0.939.
- **E2E throughput**: projected 1000-image batch under 30 minutes (the contract budget). If over, flag which component is the bottleneck (detector or classifier).

If any gate fails, surface the specifics. Do NOT try to fix anything Pi-side — Mac-side authors fixes.

## Stop after reporting

No commits, no Phase 5 work. The Mac-side agent will package the Anand-facing summary from your numbers.

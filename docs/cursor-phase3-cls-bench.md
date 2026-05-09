# Cursor megaprompt — Phase 3 disease classifier benchmark on Pi

Use this **after** the disease classifier is trained in Colab, exported to NCNN on Mac, and published as a GitHub Release. It mirrors `docs/cursor-phase2-bench.md` but for the cls model.

Pre-conditions on the Pi:
- Repo cloned and synced.
- `.venv` already primed from Phase 2 (`ultralytics`, `ncnn`).
- Disease crops val set present at `data/disease_crops/val/` — author the megaprompt below to pull this from a release if not.

---

You are the Pi-side Cursor agent. The Mac-side agent has shipped the disease classifier as GitHub Release `v0.3.0-disease`. Your job is to bench it on this Pi 5 against typical 224×224 cls inference budgets and report sanity top-1 accuracy on the val crops.

```bash
REPO=$(find / -maxdepth 6 -type d -name strawb-analysis 2>/dev/null | head -1)
cd "$REPO"
git pull --ff-only
source .venv/bin/activate

# 1. Pull the cls NCNN bundle from the release.
mkdir -p models/disease
cd models/disease
if [ ! -d yolo26n_cls_ncnn_model ]; then
  curl -L -o ncnn.tar.gz https://github.com/AKarode/strawb-analysis/releases/download/v0.3.0-disease/yolo26n_cls_ncnn_model.tar.gz
  tar -xzf ncnn.tar.gz
  rm ncnn.tar.gz
fi
ls yolo26n_cls_ncnn_model/
cd "$REPO"

# 2. Pull the val crops bundle if it isn't already on the Pi.
# (Disease crops are gitignored. The Mac-side ships them as a release asset
# alongside the cls weights so accuracy can be measured on the Pi.)
mkdir -p data/disease_crops
if [ ! -d data/disease_crops/val ]; then
  curl -L -o /tmp/val_crops.tar.gz https://github.com/AKarode/strawb-analysis/releases/download/v0.3.0-disease/disease_crops_val.tar.gz
  tar -xzf /tmp/val_crops.tar.gz -C data/disease_crops/
  rm /tmp/val_crops.tar.gz
fi
echo "val crops total: $(find data/disease_crops/val -name '*.jpg' | wc -l)"
echo "classes: $(ls data/disease_crops/val/)"

# 3. Run the cls bench.
echo "=== pre-bench ==="
vcgencmd measure_temp
vcgencmd get_throttled

OMP_NUM_THREADS=4 python scripts/bench_cls_ncnn.py \
  --model "$REPO/models/disease/yolo26n_cls_ncnn_model" \
  --crops "$REPO/data/disease_crops/val" \
  --warmup 10 \
  --iters 200

echo "=== post-bench ==="
vcgencmd measure_temp
vcgencmd get_throttled
```

## Report back

1. The full contents of `reports/cls_ncnn_bench.md`.
2. The trailing JSON summary block from stdout.
3. Any warnings or failures.

## Pass criteria

Two independent gates:

- **Latency**: cls @ 224×224 on Pi 5 NCNN should be in the 10–30 ms/image range (much faster than detect because of the smaller imgsz). Mean over 200 iters within ±20% of any consistent value in that band → pass.
- **Accuracy**: top-1 over the bench iters should be close to the val top-1 reported during Colab training (within a couple of percent). Big drop → suspect NCNN export issue or label-order mismatch.

If either gate fails, surface the specifics; don't try to fix anything Pi-side.

## Stop after reporting

No commits, no Phase 4 work (still Mac-authored). The Mac-side agent will take your numbers and decide whether to proceed to integrated CPU pipeline (`src/run_inference.py`) or revisit the classifier export.

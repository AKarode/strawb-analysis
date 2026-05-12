# Cursor megaprompt — real 1000-image E2E batch on Pi (Anand-facing measurement)

The earlier Phase 3 bench (`docs/cursor-phase3-cls-bench.md`) measured 159 Zenodo val images and projected to 1000. Anand needs the real measured number, not a projection — this run captures that.

Pre-conditions on the Pi:
- All Phase 3 setup from the previous megaprompt is in place (cls NCNN bundle at `models/disease/yolo26n_cls_ncnn_model/`, detector at `models/detect/yolo26n_zenodo_ncnn_model/`, venv active).
- Zenodo at `data/zenodo/strawberries/{training,validation}/` (813 .jpg total).
- Roboflow matt-lucky at `data/roboflow/matt-lucky-ripeness/` (≥2,000 .jpg).

---

You are the Pi-side Cursor agent. The Mac-side wants a true measured 1000-image batch time, not the 159 → 1000 projection from the prior bench. Build a 1000-image batch from real strawberry field images and run the integrated CPU pipeline. Do this in one shot.

## Step 1 — Sync + activate

```bash
REPO=$(find / -maxdepth 6 -type d -name strawb-analysis 2>/dev/null | head -1)
cd "$REPO"
git pull --ff-only
source .venv/bin/activate
```

## Step 2 — Assemble a 1000-image batch (symlinks, no copying)

Use all 813 Zenodo train+val (the domain the detector was trained for) + 187 matt-lucky to hit exactly 1000. Represents the client's actual data shape better than any single source.

```bash
BATCH=/tmp/anand_1k_batch
rm -rf "$BATCH" && mkdir -p "$BATCH"

# 1. All 813 Zenodo (train + val) — the detector's home domain.
i=0
for f in $(find "$REPO/data/zenodo/strawberries/training" \
                "$REPO/data/zenodo/strawberries/validation" \
                -name '*.jpg' 2>/dev/null | sort); do
  i=$((i+1))
  ln -s "$f" "$BATCH/zen_$(printf '%04d' $i).jpg"
done
echo "zenodo symlinks: $i"

# 2. Top up to 1000 with matt-lucky (different photographer, same domain).
need=$((1000 - i))
j=0
for f in $(find "$REPO/data/roboflow/matt-lucky-ripeness" -name '*.jpg' 2>/dev/null | sort | head -"$need"); do
  j=$((j+1))
  ln -s "$f" "$BATCH/ml_$(printf '%04d' $j).jpg"
done
echo "matt-lucky symlinks: $j"

# Sanity
N=$(find "$BATCH" -maxdepth 1 -name '*.jpg' | wc -l)
echo "batch size: $N (target 1000)"
[ "$N" -lt 999 ] && { echo "INSUFFICIENT BATCH"; exit 1; }
```

## Step 3 — Pre-run environment

```bash
echo "=== pre-run env ==="
vcgencmd measure_temp
vcgencmd get_throttled
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
date
```

## Step 4 — Run the real 1000-image batch

```bash
mkdir -p reports
time python -m src.run_inference \
  --backend cpu \
  --images "$BATCH" \
  --detector "$REPO/models/detect/yolo26n_zenodo_ncnn_model" \
  --classifier "$REPO/models/disease/yolo26n_cls_ncnn_model" \
  --out reports/inference_cpu_1k.csv

echo "=== post-run env ==="
vcgencmd measure_temp
vcgencmd get_throttled
date
```

## Step 5 — Aggregate

```bash
python - <<'PY'
import csv, statistics
rows = list(csv.DictReader(open('reports/inference_cpu_1k.csv')))
det = [float(r['detect_ms']) for r in rows if r.get('detect_ms')]
cls = [float(r['classify_ms']) for r in rows if r.get('classify_ms')]
tot = [float(r['total_ms']) for r in rows if r.get('total_ms')]
nfr = [int(r['n_total']) for r in rows if r.get('n_total')]

def stat(x, label):
    if not x: print(f"{label}: no data"); return
    xs = sorted(x)
    print(f"{label:12s} n={len(x):4d}  mean={statistics.fmean(x):8.1f} ms"
          f"  p50={xs[len(xs)//2]:8.1f}  p95={xs[int(len(xs)*0.95)-1]:8.1f}"
          f"  min={min(x):.1f}  max={max(x):.1f}")

print(f"images processed: {len(rows)}")
stat(det, "detect")
stat(cls, "classify")
stat(tot, "total")
batch_s = sum(tot) / 1000
print(f"\nbatch wall time (sum of per-image total_ms): {batch_s:.1f}s = {batch_s/60:.2f} min")
print(f"compare to projection from 159-image run: 8.0 min")
if nfr:
    print(f"fruits per image: mean {statistics.fmean(nfr):.1f}, p50 {sorted(nfr)[len(nfr)//2]}, max {max(nfr)}")

# Thermal-drift check: did per-image total trend up across the batch?
windows = []
for i in range(0, len(tot), max(1, len(tot)//10)):
    chunk = tot[i:i + max(1, len(tot)//10)]
    if chunk: windows.append(statistics.fmean(chunk))
print(f"\nper-image total_ms across 10 windows (left = early, right = late):")
print("  " + "  ".join(f"{w:.0f}" for w in windows))
if len(windows) >= 4:
    early = statistics.fmean(windows[:len(windows)//4])
    late  = statistics.fmean(windows[-len(windows)//4:])
    drift = (late - early) / early * 100
    print(f"early-vs-late drift: {drift:+.1f}% (positive = slower over time, suggests thermal)")
PY
```

## Report back

1. The `time` block from Step 4 (real wall-clock).
2. The full output of the Step 5 aggregator — especially the **batch wall time in minutes**, the early/late drift number, and whether `vcgencmd get_throttled` changed during the run.
3. Pre vs post temp.

## Pass criteria

- Measured 1000-image batch under 30 minutes (the contract budget).
- Early-vs-late drift under +20% (above that suggests thermal throttling is meaningful and we should add cooling guidance for the deploy).

## Stop after reporting

No commits. Mac-side will fold the real measured number into the Anand summary.

# Cursor megaprompt — Phase 4 count-accuracy evaluator on Pi

Tiny run. The Phase 3 + 4 megaprompts already produced `reports/inference_cpu_e2e.csv` on the Pi (159 Zenodo val images, predicted counts per image). This invocation compares those predictions to Zenodo's ground-truth annotations and emits MAE / exact-match / Pearson correlation for total / ripe / unripe counts — the rigorous "how good is the count?" answer the deliverable spec calls for.

Pre-conditions on the Pi:
- `reports/inference_cpu_e2e.csv` present from the earlier bench (the cls + E2E megaprompt step 5).
- `data/zenodo/strawberries/validation/` populated with .jpg + matching .txt YOLO label sidecars (Phase 1).

---

You are the Pi-side Cursor agent. Run the Phase 4 evaluator against the existing predictions CSV and report.

```bash
REPO=$(find / -maxdepth 6 -type d -name strawb-analysis 2>/dev/null | head -1)
cd "$REPO"
git pull --ff-only
source .venv/bin/activate

# Sanity-check inputs
ls -la reports/inference_cpu_e2e.csv
ls data/zenodo/strawberries/validation/ | head -5
echo "label sidecars: $(find data/zenodo/strawberries/validation -name '*.txt' | wc -l)"

# Run the evaluator (no extra dependencies — stdlib only)
python -m src.evaluator \
  --predictions reports/inference_cpu_e2e.csv \
  --ground-truth-images data/zenodo/strawberries/validation \
  --out reports/eval_cpu.json

# Echo the JSON for the report-back
cat reports/eval_cpu.json | python -m json.tool | head -40

# A quick interpretation block — what do these numbers actually mean?
python - <<'PY'
import json
e = json.load(open('reports/eval_cpu.json'))
print(f"\nimages evaluated: {e['n_images']}")
for k, label in [('n_total','total fruit/peduncle'),
                 ('n_ripe','ripe fruit'),
                 ('n_unripe','unripe fruit')]:
    m = e['count_metrics'][k]
    print(f"\n{label}:")
    print(f"  MAE per image: {m['mae']:.2f}  (lower=better; this is avg |gt - pred| per image)")
    print(f"  Exact match: {m['exact_match_rate']*100:.1f}%  (% of images where count was exactly right)")
    print(f"  Pearson: {m['pearson']:.3f}  (correlation between gt and pred counts; >0.9 = strong)")
    print(f"  GT mean: {m['gt_mean']:.2f}  Pred mean: {m['pred_mean']:.2f}  "
          f"({'over' if m['pred_mean'] > m['gt_mean'] else 'under'}-predicts on average)")
PY
```

## Report back

1. The trailing `python - <<'PY' ... PY` interpretation block (the human-readable summary).
2. The first 40 lines of `reports/eval_cpu.json` (so we can verify the per-class metrics).
3. Anything weird — e.g. if `gt_total` ends up 0 for most images, it means label files weren't found and we need to debug paths.

## Pass criteria

- Pearson on n_ripe ≥ 0.85 — strong linear relationship between detected and actual ripe count.
- MAE on n_total under 2 fruits/image — count accurate to within a couple of fruits in the typical case.

## Stop after reporting

No commits. Mac-side will fold the count-accuracy numbers into the Anand summary.

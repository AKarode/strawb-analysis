"""Phase 4 — accuracy comparison against ground truth.

Reads a prediction CSV produced by `src.run_inference` and compares per-image
counts (n_total, n_ripe, n_unripe) against the ground truth in the original
dataset labels. Today this only supports the Zenodo schema (3-class YOLO
.txt sidecars at <images_dir>/<stem>.txt).

Disease accuracy is intentionally not computed at the per-image level here:
the disease classifier produces a per-detection prediction, and the project's
ground truth (Kaggle Afzaal LabelMe, Zenodo bbox) doesn't co-locate disease
labels with ripeness on the same image. Disease accuracy is measured directly
on the disease classifier's val set during Phase 3 training.

Usage:

    python -m src.evaluator \\
        --predictions reports/inference_cpu.csv \\
        --ground-truth-images data/zenodo/strawberries/validation \\
        --out reports/eval_cpu.json

Output JSON shape:

    {
      "n_images": 159,
      "count_metrics": {
        "n_total":  {"mae": 0.42, "exact_match_rate": 0.81, "pearson": 0.96},
        "n_ripe":   {...},
        "n_unripe": {...}
      },
      "image_breakdown": [{"image": "...", "gt_total": 4, "pred_total": 5, ...}, ...]
    }
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from statistics import fmean

REPO_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger("evaluator")
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Zenodo class index → canonical name. Matches scripts/train_detect.py.
ZENODO_CLASS_NAMES = {0: "ripe", 1: "unripe", 2: "peduncle"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 4 — count-accuracy vs Zenodo ground truth.",
    )
    p.add_argument(
        "--predictions", type=Path, required=True,
        help="CSV produced by src.run_inference.",
    )
    p.add_argument(
        "--ground-truth-images", type=Path, required=True,
        help="Directory containing Zenodo .jpg + .txt sidecar pairs.",
    )
    p.add_argument(
        "--out", type=Path, required=True,
        help="Output JSON path.",
    )
    return p.parse_args()


def _read_zenodo_label(txt_path: Path) -> dict[str, int]:
    """Return per-class counts from a Zenodo YOLO label file."""
    counts: dict[str, int] = {"ripe": 0, "unripe": 0, "peduncle": 0}
    if not txt_path.exists():
        return counts
    for line in txt_path.read_text(encoding="utf-8", errors="replace").splitlines():
        tokens = line.strip().split()
        if not tokens:
            continue
        try:
            idx = int(tokens[0])
        except ValueError:
            continue
        name = ZENODO_CLASS_NAMES.get(idx)
        if name is not None:
            counts[name] += 1
    return counts


def _read_predictions(csv_path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    return rows


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation, or 0.0 if undefined."""
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mx, my = fmean(xs), fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def _metrics(gt: list[int], pred: list[int]) -> dict:
    if not gt:
        return {"mae": 0.0, "exact_match_rate": 0.0, "pearson": 0.0,
                "gt_mean": 0.0, "pred_mean": 0.0}
    diffs = [abs(g - p) for g, p in zip(gt, pred)]
    matches = sum(1 for g, p in zip(gt, pred) if g == p)
    return {
        "mae":              fmean(diffs),
        "exact_match_rate": matches / len(gt),
        "pearson":          _pearson([float(x) for x in gt],
                                     [float(x) for x in pred]),
        "gt_mean":          fmean(gt),
        "pred_mean":        fmean(pred),
    }


def main() -> int:
    args = parse_args()

    if not args.predictions.exists():
        print(f"predictions CSV not found: {args.predictions}", file=sys.stderr)
        return 1
    if not args.ground_truth_images.is_dir():
        print(f"ground truth dir not found: {args.ground_truth_images}",
              file=sys.stderr)
        return 1

    rows = _read_predictions(args.predictions)
    if not rows:
        print(f"prediction CSV is empty: {args.predictions}", file=sys.stderr)
        return 1

    image_breakdown: list[dict] = []
    gt_total: list[int] = []
    pr_total: list[int] = []
    gt_ripe: list[int] = []
    pr_ripe: list[int] = []
    gt_unripe: list[int] = []
    pr_unripe: list[int] = []

    for row in rows:
        img_path = Path(row["image_path"])
        # row image_path may be repo-relative; resolve against repo root.
        if not img_path.is_absolute():
            img_path = REPO_ROOT / img_path
        # Try matching by basename under ground-truth dir if direct path fails.
        txt_path = img_path.with_suffix(".txt")
        if not txt_path.exists():
            cand = list(args.ground_truth_images.rglob(img_path.stem + ".txt"))
            if cand:
                txt_path = cand[0]

        gt = _read_zenodo_label(txt_path)
        pr = {
            "ripe":     int(row.get("n_ripe", 0) or 0),
            "unripe":   int(row.get("n_unripe", 0) or 0),
            "peduncle": int(row.get("n_peduncle", 0) or 0),
        }
        gt_t = sum(gt.values())
        pr_t = int(row.get("n_total", 0) or 0)

        image_breakdown.append({
            "image":     str(img_path.name),
            "gt_total":  gt_t,
            "pred_total": pr_t,
            "gt_ripe":   gt["ripe"],
            "pred_ripe": pr["ripe"],
            "gt_unripe":   gt["unripe"],
            "pred_unripe": pr["unripe"],
            "diseases_present": row.get("diseases_present", ""),
        })

        gt_total.append(gt_t);   pr_total.append(pr_t)
        gt_ripe.append(gt["ripe"]);     pr_ripe.append(pr["ripe"])
        gt_unripe.append(gt["unripe"]); pr_unripe.append(pr["unripe"])

    out = {
        "n_images":     len(rows),
        "predictions":  str(args.predictions),
        "ground_truth": str(args.ground_truth_images),
        "count_metrics": {
            "n_total":  _metrics(gt_total,  pr_total),
            "n_ripe":   _metrics(gt_ripe,   pr_ripe),
            "n_unripe": _metrics(gt_unripe, pr_unripe),
        },
        "image_breakdown": image_breakdown,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))

    cm = out["count_metrics"]
    logger.info("[done] %s", args.out)
    logger.info("  n_images: %d", out["n_images"])
    for k in ("n_total", "n_ripe", "n_unripe"):
        m = cm[k]
        logger.info("  %-9s mae=%.2f  exact=%.2f  pearson=%.3f",
                    k, m["mae"], m["exact_match_rate"], m["pearson"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

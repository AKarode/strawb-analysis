#!/usr/bin/env python3
"""Evaluate the disease classifier on a held-out test split.

Walks `data/disease_crops/<split>/<class>/<crop>.jpg`, runs the model
on each crop, and computes per-class top-1 / top-5 plus a confusion
matrix. Writes a per-image CSV and a summary JSON to `reports/`.

Usage:

    python scripts/eval_disease_test.py \\
        --weights models/disease/yolo26n_cls.pt \\
        --data-root data/disease_crops \\
        --split test \\
        --out reports/disease_test_eval

Outputs:

    reports/disease_test_eval.csv   — one row per crop
    reports/disease_test_eval.json  — summary metrics

The model may have been trained on classes the eval split doesn't
contain (e.g. `healthy` is sourced from Roboflow and may be absent in
a Kaggle-only build). The script handles that gracefully — missing
classes are reported but don't fail the run.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate disease classifier on a held-out split.")
    p.add_argument("--weights", type=Path, required=True,
                   help="Path to trained .pt classifier.")
    p.add_argument("--data-root", type=Path,
                   default=Path("data/disease_crops"),
                   help="Root containing <split>/<class>/<crop>.jpg.")
    p.add_argument("--split", default="test",
                   help="Which subdir of data-root to evaluate.")
    p.add_argument("--imgsz", type=int, default=224)
    p.add_argument("--device", default=None,
                   help='Torch device: "cpu", "mps", or "0" (CUDA 0).')
    p.add_argument("--out", type=Path,
                   default=Path("reports/disease_test_eval"),
                   help="Output stem; produces <out>.csv and <out>.json.")
    p.add_argument("--batch", type=int, default=32,
                   help="Batch size for model.predict() calls.")
    p.add_argument("--task", default="classify",
                   help="Force YOLO task (needed for NCNN bundles, which "
                        "ultralytics dispatches to detect by default).")
    return p.parse_args()


def discover_crops(split_root: Path) -> list[tuple[str, Path]]:
    """Return [(true_class, image_path), ...] in stable sorted order."""
    rows: list[tuple[str, Path]] = []
    for cls_dir in sorted(split_root.iterdir()):
        if not cls_dir.is_dir():
            continue
        for img in sorted(cls_dir.glob("*.jpg")):
            rows.append((cls_dir.name, img))
    return rows


def main() -> int:
    args = parse_args()
    split_root = args.data_root / args.split
    if not split_root.is_dir():
        print(f"split not found: {split_root}", file=sys.stderr)
        return 1

    crops = discover_crops(split_root)
    if not crops:
        print(f"no crops under {split_root}", file=sys.stderr)
        return 1
    print(f"[eval] {len(crops)} crops across {len({c for c,_ in crops})} classes "
          f"in {split_root}", flush=True)

    from ultralytics import YOLO

    model = YOLO(str(args.weights), task=args.task)
    # Trained class set (sorted index → name). For a YOLO cls model this
    # comes from model.names: {idx: name}.
    model_names: dict[int, str] = model.names if isinstance(model.names, dict) \
        else {i: n for i, n in enumerate(model.names)}
    print(f"[eval] model classes ({len(model_names)}): "
          f"{[model_names[i] for i in sorted(model_names)]}", flush=True)

    # Per-row predictions and aggregates.
    out_csv = args.out.with_suffix(".csv")
    out_json = args.out.with_suffix(".json")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    top1_hits = 0
    top5_hits = 0
    per_class_total: Counter[str] = Counter()
    per_class_top1: Counter[str] = Counter()
    per_class_top5: Counter[str] = Counter()
    # confusion[true][pred] = count
    confusion: dict[str, Counter[str]] = defaultdict(Counter)

    eval_start = time.time()
    with out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "path", "true", "pred_top1", "pred_top1_conf",
            "pred_top5", "pred_top5_confs", "correct_top1", "correct_top5",
        ])

        # Batched predict to avoid one Python call per image. ultralytics
        # accepts a list of paths.
        for start in range(0, len(crops), args.batch):
            chunk = crops[start:start + args.batch]
            paths = [str(p) for _, p in chunk]
            results = model.predict(
                paths,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )
            for (true_cls, img_path), r in zip(chunk, results):
                probs = r.probs
                if probs is None:
                    continue
                top5_idx = probs.top5
                top5_conf = [float(c) for c in probs.top5conf.tolist()]
                top5_names = [model_names[int(i)] for i in top5_idx]
                top1_name = top5_names[0]
                top1_conf = top5_conf[0]

                correct1 = (top1_name == true_cls)
                correct5 = (true_cls in top5_names)

                w.writerow([
                    str(img_path.relative_to(args.data_root.parent)
                        if args.data_root.parent in img_path.parents
                        else img_path),
                    true_cls,
                    top1_name,
                    f"{top1_conf:.4f}",
                    "|".join(top5_names),
                    "|".join(f"{c:.4f}" for c in top5_conf),
                    int(correct1),
                    int(correct5),
                ])

                total += 1
                top1_hits += int(correct1)
                top5_hits += int(correct5)
                per_class_total[true_cls] += 1
                per_class_top1[true_cls] += int(correct1)
                per_class_top5[true_cls] += int(correct5)
                confusion[true_cls][top1_name] += 1

            done = min(start + args.batch, len(crops))
            if done % 256 == 0 or done == len(crops):
                print(f"  {done}/{len(crops)} crops "
                      f"({top1_hits/total*100:.2f}% top-1 so far)",
                      flush=True)

    elapsed = time.time() - eval_start

    # Build summary.
    per_class = []
    for cls in sorted(per_class_total):
        n = per_class_total[cls]
        per_class.append({
            "class": cls,
            "n": n,
            "top1": round(per_class_top1[cls] / n, 4),
            "top5": round(per_class_top5[cls] / n, 4),
        })

    # Missing classes the model knows about but eval split lacks.
    eval_classes = set(per_class_total)
    missing = sorted(set(model_names.values()) - eval_classes)

    summary = {
        "weights": str(args.weights),
        "data_root": str(args.data_root),
        "split": args.split,
        "imgsz": args.imgsz,
        "n_images": total,
        "n_classes_eval": len(eval_classes),
        "n_classes_model": len(model_names),
        "missing_in_eval": missing,
        "top1": round(top1_hits / total, 4),
        "top5": round(top5_hits / total, 4),
        "per_class": per_class,
        "confusion_matrix": {
            t: dict(confusion[t]) for t in sorted(confusion)
        },
        "elapsed_sec": round(elapsed, 2),
    }

    out_json.write_text(json.dumps(summary, indent=2))

    # Pretty-print summary.
    print()
    print(f"[eval] {total} images in {elapsed:.1f}s "
          f"({total/elapsed:.1f} img/s)")
    print(f"[eval] overall top-1: {summary['top1']:.4f}   "
          f"top-5: {summary['top5']:.4f}")
    if missing:
        print(f"[eval] classes in model but missing from eval split: {missing}")
    print()
    print(f"  {'class':<26s} {'n':>5s}  {'top1':>6s}  {'top5':>6s}")
    print(f"  {'-'*26} {'-'*5}  {'-'*6}  {'-'*6}")
    for r in per_class:
        print(f"  {r['class']:<26s} {r['n']:>5d}  {r['top1']:>6.4f}  {r['top5']:>6.4f}")
    print()
    print(f"[eval] wrote {out_csv} ({total} rows)")
    print(f"[eval] wrote {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

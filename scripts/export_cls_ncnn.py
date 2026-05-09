#!/usr/bin/env python3
"""Export a trained YOLO26-cls disease classifier to NCNN.

Phase 3 → Phase 4 handoff. Pi 5 CPU classifier path uses NCNN FP32 @ 224.
This is the Mac-side counterpart to scripts/export_detect_ncnn.py.

Usage:
    python scripts/export_cls_ncnn.py \\
        --weights models/disease/yolo26n_cls.pt

Output (default --imgsz 224):
    models/disease/yolo26n_cls_ncnn_model/
        model.ncnn.param
        model.ncnn.bin
        metadata.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export a trained YOLO26-cls model to NCNN.",
    )
    p.add_argument(
        "--weights", type=Path, required=True,
        help="Path to the trained .pt (e.g. models/disease/yolo26n_cls.pt).",
    )
    p.add_argument(
        "--imgsz", type=int, default=224,
        help="Inference resolution baked into the export. Cls default: 224.",
    )
    p.add_argument(
        "--half", action="store_true",
        help="FP16 export. Default FP32 — Pi 5 NCNN does not benefit from FP16.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.weights.exists():
        print(f"weights not found: {args.weights}", file=sys.stderr)
        return 1

    from ultralytics import YOLO

    print(f"[load] {args.weights}", flush=True)
    model = YOLO(str(args.weights))
    print(f"[load] task={model.task} | nc={len(model.names)} | "
          f"names={list(model.names.values())}", flush=True)

    if model.task != "classify":
        print(f"[fail] expected classification model, got task={model.task}",
              file=sys.stderr)
        return 1

    print(f"[export] format=ncnn imgsz={args.imgsz} half={args.half}",
          flush=True)
    out = model.export(format="ncnn", imgsz=args.imgsz, half=args.half)
    out_path = Path(out).resolve()
    print(f"[done] {out_path}", flush=True)

    expected = {"model.ncnn.param", "model.ncnn.bin", "metadata.yaml"}
    present = {p.name for p in out_path.iterdir() if p.is_file()} \
        if out_path.is_dir() else set()
    missing = expected - present
    if missing:
        print(f"[warn] missing expected files in bundle: {sorted(missing)}",
              file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())

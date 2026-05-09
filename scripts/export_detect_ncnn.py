#!/usr/bin/env python3
"""Export a trained YOLO26 detector to NCNN.

Phase 2 → Phase 4 handoff. The Pi 5 CPU backend uses NCNN FP32 @ 640×640
(see .planning/PROJECT.md — INT8 on NCNN is not viable on Pi 5 as of
April 2026). This script is the single Mac-side step that turns a trained
.pt into the NCNN bundle the Pi consumes.

Usage:
    python scripts/export_detect_ncnn.py \\
        --weights models/detect/yolo26n_zenodo.pt

Output (default --imgsz 640):
    models/detect/yolo26n_zenodo_ncnn_model/
        model.ncnn.param
        model.ncnn.bin
        metadata.yaml

Defaults match the Pi runtime contract — only override if you know why.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export a trained YOLO26 detector to NCNN.",
    )
    p.add_argument(
        "--weights", type=Path, required=True,
        help="Path to the trained .pt (e.g. models/detect/yolo26n_zenodo.pt).",
    )
    p.add_argument(
        "--imgsz", type=int, default=640,
        help="Inference resolution baked into the export. Pi target: 640.",
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

    if model.task != "detect":
        print(f"[fail] expected detection model, got task={model.task}",
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

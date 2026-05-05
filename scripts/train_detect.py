#!/usr/bin/env python3
"""Train YOLO26n detection on the Zenodo strawberry dataset.

Phase 2 of the Strawberry Vision Pi roadmap. Detection target is the 3-class
schema from .planning/PROJECT.md: ripe / unripe / peduncle.

Usage:
    python scripts/train_detect.py \\
        --data-root data/zenodo/strawberries \\
        --epochs 100 \\
        --imgsz 640 \\
        --batch 16

The data root must contain training/ and validation/ subdirs, each with paired
.jpg and YOLO-format .txt labels. This is the layout produced by extracting
the Zenodo record 6126677 archive.

Environments tested:
    - Colab T4 / A100 / L4 (CUDA)            ← recommended for full training
    - macOS Apple Silicon (MPS)              ← fast smoke, slow for full
    - Linux + CUDA

Outputs (under --project / --name):
    weights/best.pt   — best val-mAP epoch
    weights/last.pt   — final epoch
    results.csv       — per-epoch metrics
    results.png       — loss + metric curves
    confusion_matrix*.png
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent

# Class index order matches the upstream Zenodo strawberries.yaml.
# If the upstream yaml ever disagrees, training labels and this list will
# desync silently — train_detect.py prints both at startup so a mismatch is
# obvious in the run log.
CLASS_NAMES = {0: "ripe", 1: "unripe", 2: "peduncle"}


def materialize_data_yaml(data_root: Path, dest: Path) -> Path:
    """Write an Ultralytics-compatible data yaml with absolute paths.

    Ultralytics resolves `path` against its configured datasets root, which
    varies by environment — using an absolute path here makes the yaml
    portable between Mac, Colab, and Linux without touching settings.yaml.
    """
    abs_root = data_root.resolve()
    cfg = {
        "path": str(abs_root),
        "train": "training",
        "val": "validation",
        "names": CLASS_NAMES,
    }
    dest.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return dest


def cross_check_upstream_yaml(data_root: Path) -> None:
    """Print the upstream strawberries.yaml class order, if present, so a
    silent mismatch with CLASS_NAMES is obvious in the training log.
    """
    upstream = data_root / "strawberries.yaml"
    if not upstream.exists():
        upstream = data_root / "data.yaml"
    if not upstream.exists():
        print("[note] no upstream class yaml found at "
              f"{data_root}/strawberries.yaml or data.yaml — "
              "trusting CLASS_NAMES.", flush=True)
        return
    try:
        doc = yaml.safe_load(upstream.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"[note] upstream yaml unreadable ({upstream}): {exc}", flush=True)
        return
    upstream_names = doc.get("names")
    print(f"[check] upstream yaml: {upstream}", flush=True)
    print(f"[check] upstream names: {upstream_names}", flush=True)
    print(f"[check] script  names: {CLASS_NAMES}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train YOLO26n detection on Zenodo strawberries.",
    )
    p.add_argument(
        "--data-root", type=Path, required=True,
        help="Path to data/zenodo/strawberries (contains training/ and validation/).",
    )
    p.add_argument(
        "--weights", default="yolo26n.pt",
        help="Pretrained weights (file path or Ultralytics model name). "
             "Auto-downloaded if not on disk.",
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument(
        "--device", default=None,
        help='Torch device. Examples: "0" (CUDA gpu 0), "cpu", "mps". '
             "If unset, Ultralytics auto-detects.",
    )
    p.add_argument(
        "--project", default="runs/detect",
        help="Ultralytics project dir. On Colab, point this at "
             "/content/drive/MyDrive/... so checkpoints survive disconnect.",
    )
    p.add_argument("--name", default="zenodo_yolo26n")
    p.add_argument(
        "--patience", type=int, default=20,
        help="Early-stopping patience (epochs without val improvement).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--resume", action="store_true",
        help="Resume from last.pt in the project/name dir if present.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.data_root.exists():
        print(f"data root not found: {args.data_root}", file=sys.stderr)
        return 1
    for sub in ("training", "validation"):
        if not (args.data_root / sub).is_dir():
            print(f"missing {args.data_root / sub}", file=sys.stderr)
            return 1

    cross_check_upstream_yaml(args.data_root)

    from ultralytics import YOLO  # imported here so --help works without it

    with tempfile.TemporaryDirectory() as td:
        data_yaml = materialize_data_yaml(args.data_root, Path(td) / "zenodo.yaml")
        print(f"[data] {data_yaml.read_text()}", flush=True)

        model = YOLO(args.weights)
        model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=args.name,
            patience=args.patience,
            seed=args.seed,
            resume=args.resume,
            plots=True,
            exist_ok=args.resume,
        )

    print(f"[done] best weights: {args.project}/{args.name}/weights/best.pt", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

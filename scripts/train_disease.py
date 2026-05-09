#!/usr/bin/env python3
"""Train YOLO26n-cls on disease crops.

Phase 3 of the Strawberry Vision Pi roadmap. The disease classifier runs
on per-fruit crops emitted by the detector at inference time. Training
inputs must look like inference inputs, so the dataset is the
`data/disease_crops/` folder layout produced by
`scripts/build_disease_crops.py`:

    data/disease_crops/
        train/<class>/<crop>.jpg
        val/<class>/<crop>.jpg
        test/<class>/<crop>.jpg

Usage:

    python scripts/train_disease.py \\
        --data-root data/disease_crops \\
        --weights yolo26n-cls.pt \\
        --epochs 50 \\
        --imgsz 224 \\
        --batch 64

Environments tested:
    - Colab T4 / A100 / L4 (CUDA)            ← recommended for full training
    - macOS Apple Silicon (MPS)              ← fast smoke
    - Linux + CUDA

Outputs (under --project / --name):
    weights/best.pt   — best val top-1 epoch
    weights/last.pt   — final epoch
    results.csv       — per-epoch metrics
    results.png       — loss + accuracy curves
    confusion_matrix*.png

Target accuracy: ≥ 90% top-1 on val. Baseline to beat: BrunoKreiner's
92–93% mAP50 (YOLOv8-XL seg, different metric, 2023) on the same Kaggle
disease dataset — our metric is classification top-1 on tight crops, so
not a one-to-one comparison but the right ballpark.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent

# 8-class universe matches DISEASE_CLASSES in src/manifest/schema.py. The
# trainer reads class folder names off disk; this list is documentation
# only — Ultralytics infers class indexes from sorted dir names.
EXPECTED_CLASSES = sorted([
    "angular_leafspot",
    "anthracnose_fruit_rot",
    "blossom_blight",
    "gray_mold",
    "healthy",
    "leaf_spot",
    "powdery_mildew_fruit",
    "powdery_mildew_leaf",
])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train YOLO26n-cls on disease crops.",
    )
    p.add_argument(
        "--data-root", type=Path, required=True,
        help="Path to data/disease_crops (contains train/, val/, optional test/).",
    )
    p.add_argument(
        "--weights", default="yolo26n-cls.pt",
        help="Pretrained classification weights. Auto-downloaded.",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=224)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument(
        "--device", default=None,
        help='Torch device. Examples: "0" (CUDA 0), "cpu", "mps".',
    )
    p.add_argument(
        "--project", default="runs/classify",
        help="Ultralytics project dir. On Colab, point at "
             "/content/drive/MyDrive/... so checkpoints survive disconnect.",
    )
    p.add_argument("--name", default="disease_yolo26n_cls")
    p.add_argument(
        "--patience", type=int, default=15,
        help="Early-stopping patience (epochs without val improvement).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--resume", action="store_true",
        help="Resume from last.pt in project/name dir if present.",
    )
    return p.parse_args()


def cross_check_class_dirs(data_root: Path) -> None:
    """Print the train/ class directories so a missing/extra class is
    obvious in the run log before training starts.
    """
    train_dir = data_root / "train"
    if not train_dir.is_dir():
        return
    found = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
    missing = [c for c in EXPECTED_CLASSES if c not in found]
    extra = [c for c in found if c not in EXPECTED_CLASSES]
    print(f"[check] train/ classes ({len(found)}): {found}", flush=True)
    if missing:
        print(f"[check] MISSING expected classes: {missing}", flush=True)
    if extra:
        print(f"[check] EXTRA classes (not in EXPECTED_CLASSES): {extra}", flush=True)


def main() -> int:
    args = parse_args()

    if not args.data_root.exists():
        print(f"data root not found: {args.data_root}", file=sys.stderr)
        return 1
    for sub in ("train", "val"):
        if not (args.data_root / sub).is_dir():
            print(f"missing {args.data_root / sub} (need train/ and val/)",
                  file=sys.stderr)
            return 1

    cross_check_class_dirs(args.data_root)

    from ultralytics import YOLO

    model = YOLO(args.weights)
    model.train(
        data=str(args.data_root.resolve()),
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

    print(f"[done] best weights: {args.project}/{args.name}/weights/best.pt",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

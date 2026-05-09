#!/usr/bin/env python3
"""Build the disease classifier training set from on-disk source datasets.

Reads two source datasets and emits a folder layout consumable by
ultralytics' classification trainer:

    data/disease_crops/{train,val,test}/{class_name}/{src}__{idx}.jpg

8 canonical classes (project_schema_class strings from src.manifest.schema):

    angular_leafspot         (kaggle)
    anthracnose_fruit_rot    (kaggle)
    blossom_blight           (kaggle)
    gray_mold                (kaggle)
    leaf_spot                (kaggle)
    powdery_mildew_fruit     (kaggle)
    powdery_mildew_leaf      (kaggle)
    healthy                  (roboflow research-proj-disease, 'Healthy Fruit' only)

Why two sources: the Kaggle Afzaal LabelMe set has every disease but no
healthy class — its scope is "diseased close-ups." The Roboflow
research-proj-disease set has a native 'Healthy Fruit' class with bbox
annotations, so we pull only those crops to fill the classifier's
null-option (ripe-but-not-diseased) bucket. The original plan to
synthesize healthy crops from non-diseased Zenodo fruits was superseded:
native healthy_fruit annotations are cleaner.

Why crops at all: the project pipeline runs detection → crop each fruit →
classify. We never feed a full field scene to the classifier. So training
inputs must look like inference inputs: tight crops around a single fruit
or leaf region.

Usage:

    python scripts/build_disease_crops.py \\
        --kaggle-root data/disease \\
        --roboflow-root data/roboflow/research-proj-disease \\
        --out data/disease_crops \\
        --pad-pct 0.10

Idempotent: existing crop files are overwritten. Deterministic ordering
(sorted globs) means re-runs produce byte-identical output if sources
don't change.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.manifest.schema import DISEASE_CLASSES  # noqa: E402

logger = logging.getLogger("build_disease_crops")
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Canonical class string (project_schema_class) -> directory name. Kept
# 1:1 today but isolating the mapping makes it easy to e.g. fold both
# powdery_mildew variants into one bucket later if class imbalance bites.
CLASS_DIRNAME = {c: c for c in DISEASE_CLASSES}

# Kaggle LabelMe shape labels -> canonical class.
_KAGGLE_LABEL_TO_CLASS = {
    "Angular Leafspot":      "angular_leafspot",
    "Anthracnose Fruit Rot": "anthracnose_fruit_rot",
    "Blossom Blight":        "blossom_blight",
    "Gray Mold":             "gray_mold",
    "Leaf Spot":             "leaf_spot",
    "Powdery Mildew Fruit":  "powdery_mildew_fruit",
    "Powdery Mildew Leaf":   "powdery_mildew_leaf",
}

# Roboflow research-proj-disease class strings (from data.yaml) -> canonical.
# We only keep crops that map to "healthy" — the disease coverage in this
# dataset overlaps with Kaggle and the Kaggle annotations are higher-fidelity.
_ROBOFLOW_LABEL_TO_CLASS = {
    "Healthy Fruit": "healthy",
}


def _bbox_from_polygon(points: list[list[float]]) -> Optional[tuple[float, float, float, float]]:
    """Return (x0, y0, x1, y1) tight bounding rect of a LabelMe polygon."""
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _pad_clip(bbox: tuple[float, float, float, float],
              w: int, h: int, pad_pct: float) -> tuple[int, int, int, int]:
    """Pad bbox by pad_pct of its own dims, then clip to [0,w]/[0,h]."""
    x0, y0, x1, y1 = bbox
    bw, bh = x1 - x0, y1 - y0
    px, py = bw * pad_pct, bh * pad_pct
    x0 = max(0, int(round(x0 - px)))
    y0 = max(0, int(round(y0 - py)))
    x1 = min(w, int(round(x1 + px)))
    y1 = min(h, int(round(y1 + py)))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None  # type: ignore[return-value]
    return x0, y0, x1, y1


def _crop_and_save(
    image_path: Path,
    bbox: tuple[int, int, int, int],
    out_path: Path,
    quality: int,
) -> bool:
    from PIL import Image
    try:
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            crop = im.crop(bbox)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(out_path, format="JPEG", quality=quality)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("crop fail %s -> %s: %s", image_path, out_path, exc)
        return False


# ---------------------------------------------------------------------------
# Kaggle Afzaal (LabelMe polygon → bbox crops)
# ---------------------------------------------------------------------------

def process_kaggle(
    kaggle_root: Path,
    out_root: Path,
    pad_pct: float,
    quality: int,
) -> Counter:
    """Walk Kaggle disease/{train,val,test}/, cut a crop per LabelMe shape."""
    counts: Counter = Counter()
    splits = [("train", "train"), ("val", "val"), ("test", "test")]

    from PIL import Image

    for src_dir, split in splits:
        d = kaggle_root / src_dir
        if not d.is_dir():
            logger.warning("kaggle: split missing: %s", d)
            continue
        for img_path in sorted(d.rglob("*.jpg")):
            json_path = img_path.with_suffix(".json")
            if not json_path.exists():
                continue
            try:
                with open(json_path, "r", encoding="utf-8") as fh:
                    doc = json.load(fh)
            except Exception as exc:  # noqa: BLE001
                logger.warning("kaggle: bad json %s: %s", json_path, exc)
                continue

            try:
                with Image.open(img_path) as im:
                    w, h = im.size
            except Exception as exc:  # noqa: BLE001
                logger.warning("kaggle: bad image %s: %s", img_path, exc)
                continue

            for idx, shape in enumerate(doc.get("shapes", []) or []):
                label = shape.get("label", "")
                canon = _KAGGLE_LABEL_TO_CLASS.get(label)
                if canon is None:
                    continue
                bbox_raw = _bbox_from_polygon(shape.get("points", []))
                if bbox_raw is None:
                    continue
                bbox = _pad_clip(bbox_raw, w, h, pad_pct)
                if bbox is None:
                    continue
                out_name = f"kaggle__{img_path.stem}__{idx:02d}.jpg"
                out_path = out_root / split / CLASS_DIRNAME[canon] / out_name
                if _crop_and_save(img_path, bbox, out_path, quality):
                    counts[(split, canon)] += 1
    return counts


# ---------------------------------------------------------------------------
# Roboflow research-proj-disease (YOLO bbox → crops, healthy_fruit only)
# ---------------------------------------------------------------------------

def _read_roboflow_classes(yaml_path: Path) -> list[str]:
    import yaml
    if not yaml_path.exists():
        return []
    try:
        with open(yaml_path, "r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
        names = doc.get("names")
        if isinstance(names, list):
            return [str(n) for n in names]
        if isinstance(names, dict):
            return [str(names[k]) for k in sorted(names)]
    except Exception as exc:  # noqa: BLE001
        logger.warning("roboflow: failed to parse %s: %s", yaml_path, exc)
    return []


def process_roboflow(
    rf_root: Path,
    out_root: Path,
    pad_pct: float,
    quality: int,
) -> Counter:
    """Walk research-proj-disease/{train,valid,test}/{images,labels}/.

    YOLO format: class_id x_center y_center w h, all normalized [0,1].
    Keep only crops whose canonical class is in _ROBOFLOW_LABEL_TO_CLASS.
    """
    counts: Counter = Counter()
    yaml_path = rf_root / "data.yaml"
    class_names = _read_roboflow_classes(yaml_path)
    if not class_names:
        logger.warning("roboflow: no class list at %s — skipping", yaml_path)
        return counts

    keep_idx_to_canon: dict[int, str] = {}
    for i, name in enumerate(class_names):
        canon = _ROBOFLOW_LABEL_TO_CLASS.get(name)
        if canon is not None:
            keep_idx_to_canon[i] = canon
    if not keep_idx_to_canon:
        logger.warning("roboflow: data.yaml has no classes mapping to {Healthy Fruit}")
        return counts

    splits = [("train", "train"), ("valid", "val"), ("test", "test")]
    from PIL import Image

    for src_dir, split in splits:
        images_dir = rf_root / src_dir / "images"
        labels_dir = rf_root / src_dir / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            logger.warning("roboflow: split missing: %s", rf_root / src_dir)
            continue
        for img_path in sorted(p for ext in ("*.jpg", "*.jpeg", "*.png")
                               for p in images_dir.rglob(ext)):
            txt_path = labels_dir / (img_path.stem + ".txt")
            if not txt_path.exists():
                continue
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
            except Exception as exc:  # noqa: BLE001
                logger.warning("roboflow: bad image %s: %s", img_path, exc)
                continue

            for idx, line in enumerate(txt_path.read_text(encoding="utf-8",
                                                          errors="replace").splitlines()):
                tokens = line.strip().split()
                if len(tokens) < 5:
                    continue
                try:
                    cls_idx = int(tokens[0])
                    cx, cy, bw, bh = (float(t) for t in tokens[1:5])
                except ValueError:
                    continue
                canon = keep_idx_to_canon.get(cls_idx)
                if canon is None:
                    continue
                # YOLO normalized → absolute pixel bbox.
                x0 = (cx - bw / 2) * w
                y0 = (cy - bh / 2) * h
                x1 = (cx + bw / 2) * w
                y1 = (cy + bh / 2) * h
                bbox = _pad_clip((x0, y0, x1, y1), w, h, pad_pct)
                if bbox is None:
                    continue
                out_name = f"roboflow__{img_path.stem}__{idx:02d}.jpg"
                out_path = out_root / split / CLASS_DIRNAME[canon] / out_name
                if _crop_and_save(img_path, bbox, out_path, quality):
                    counts[(split, canon)] += 1
    return counts


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build disease classifier training crops.",
    )
    p.add_argument("--kaggle-root", type=Path,
                   default=REPO_ROOT / "data/disease",
                   help="Kaggle Afzaal LabelMe root with train/val/test/.")
    p.add_argument("--roboflow-root", type=Path,
                   default=REPO_ROOT / "data/roboflow/research-proj-disease",
                   help="Roboflow research-proj-disease YOLO export root.")
    p.add_argument("--out", type=Path,
                   default=REPO_ROOT / "data/disease_crops",
                   help="Output root for {train,val,test}/{class}/crops.")
    p.add_argument("--pad-pct", type=float, default=0.10,
                   help="Padding around polygon/bbox as a fraction of bbox dims (0.10 = 10%%).")
    p.add_argument("--quality", type=int, default=92,
                   help="JPEG quality for output crops.")
    p.add_argument("--skip-roboflow", action="store_true",
                   help="Skip the Roboflow healthy_fruit step (Kaggle only).")
    p.add_argument("--skip-kaggle", action="store_true",
                   help="Skip the Kaggle disease step (Roboflow healthy only).")
    return p.parse_args()


def _print_counts(counts: Counter, label: str) -> None:
    if not counts:
        logger.info("[%s] no crops emitted", label)
        return
    logger.info("[%s] crops by (split, class):", label)
    by_class: dict[str, dict[str, int]] = {}
    for (split, canon), n in sorted(counts.items()):
        by_class.setdefault(canon, {})[split] = n
    for canon in sorted(by_class):
        row = by_class[canon]
        logger.info(
            "  %-26s train=%-5d val=%-5d test=%-5d",
            canon, row.get("train", 0), row.get("val", 0), row.get("test", 0),
        )


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    total: Counter = Counter()
    if not args.skip_kaggle:
        if not args.kaggle_root.is_dir():
            logger.error("kaggle root not found: %s", args.kaggle_root)
            return 1
        c = process_kaggle(args.kaggle_root, args.out, args.pad_pct, args.quality)
        _print_counts(c, "kaggle")
        total.update(c)
    if not args.skip_roboflow:
        if not args.roboflow_root.is_dir():
            logger.error("roboflow root not found: %s", args.roboflow_root)
            return 1
        c = process_roboflow(args.roboflow_root, args.out, args.pad_pct, args.quality)
        _print_counts(c, "roboflow")
        total.update(c)

    logger.info("---")
    _print_counts(total, "total")
    logger.info("output: %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

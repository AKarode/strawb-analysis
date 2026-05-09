"""Phase 4 — integrated CPU pipeline.

Two-stage inference: detector → per-fruit crop → disease classifier.
Per-image CSV output for downstream evaluation.

Usage (CPU backend, the only one implemented today):

    python -m src.run_inference \\
        --backend cpu \\
        --images data/zenodo/strawberries/validation \\
        --detector models/detect/yolo26n_zenodo_ncnn_model \\
        --classifier models/disease/yolo26n_cls_ncnn_model \\
        --out reports/inference_cpu.csv

Hailo backend (Phase 5) is out of scope here; --backend hailo currently
errors with a clear message.

CSV schema (one row per image):

    image_path, image_w, image_h,
    n_total, n_ripe, n_unripe, n_peduncle,
    diseases_present, healthy_count,
    detect_ms, classify_ms, total_ms

`diseases_present` is a semicolon-separated list of canonical disease
class names that appeared on at least one fruit in the image (e.g.
"gray_mold;leaf_spot"). Empty if all classified fruits came back
healthy or if the classifier is not provided.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger("run_inference")
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Detection classes we expand individual counts for. Anything else falls
# into n_total but not the per-class buckets.
RIPE_CLASS = "ripe"
UNRIPE_CLASS = "unripe"
PEDUNCLE_CLASS = "peduncle"

# Classifier output we treat as "no disease."
HEALTHY_CLS = "healthy"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 4 — integrated CPU pipeline.",
    )
    p.add_argument(
        "--backend", choices=("cpu", "hailo"), default="cpu",
        help='Inference backend. "cpu" uses NCNN (Phase 4). '
             '"hailo" uses HEFs via hailo-platform (Phase 5, not implemented).',
    )
    p.add_argument(
        "--images", type=Path, required=True,
        help="Directory of images to process.",
    )
    p.add_argument(
        "--detector", type=Path, required=True,
        help="Detector path. NCNN model dir for cpu backend, .hef for hailo.",
    )
    p.add_argument(
        "--classifier", type=Path, default=None,
        help="Classifier path (optional). NCNN model dir for cpu, .hef for hailo. "
             "If omitted, run_inference does detection only and skips disease cols.",
    )
    p.add_argument(
        "--out", type=Path, required=True,
        help="Output CSV path.",
    )
    p.add_argument(
        "--detect-imgsz", type=int, default=640,
        help="Detector input resolution (matches the export).",
    )
    p.add_argument(
        "--classify-imgsz", type=int, default=224,
        help="Classifier input resolution (matches the export).",
    )
    p.add_argument(
        "--conf", type=float, default=0.25,
        help="Detection confidence threshold.",
    )
    p.add_argument(
        "--iou", type=float, default=0.45,
        help="Detection NMS IoU threshold.",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Optional cap on number of images processed (for smoke tests).",
    )
    return p.parse_args()


def _list_images(images_dir: Path) -> list[Path]:
    exts = ("*.jpg", "*.jpeg", "*.png")
    out: list[Path] = []
    for ext in exts:
        out.extend(images_dir.rglob(ext))
    return sorted(out)


def _crop_and_classify(
    image_path: Path,
    detections: list[dict],
    classifier,
    classify_imgsz: int,
) -> tuple[list[Optional[str]], float]:
    """Run the classifier on each detected fruit. Returns (per-detection
    class names, total classifier ms).

    Skips peduncle detections — those aren't fruits.
    """
    from PIL import Image

    pred_names = list(classifier.names.values()) if classifier is not None else []
    classify_total_ms = 0.0
    per_det_class: list[Optional[str]] = []

    if classifier is None or not detections:
        return [None] * len(detections), 0.0

    with Image.open(image_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        for det in detections:
            if det["cls"] == PEDUNCLE_CLASS:
                per_det_class.append(None)
                continue
            x0, y0, x1, y1 = det["bbox"]
            x0 = max(0, int(round(x0))); y0 = max(0, int(round(y0)))
            x1 = min(w, int(round(x1))); y1 = min(h, int(round(y1)))
            if x1 - x0 < 4 or y1 - y0 < 4:
                per_det_class.append(None)
                continue
            crop = im.crop((x0, y0, x1, y1))
            t0 = time.perf_counter()
            res = classifier(crop, imgsz=classify_imgsz, verbose=False)
            classify_total_ms += (time.perf_counter() - t0) * 1000
            if res and res[0].probs is not None:
                top1_idx = int(res[0].probs.top1)
                per_det_class.append(pred_names[top1_idx])
            else:
                per_det_class.append(None)

    return per_det_class, classify_total_ms


def _run_detector(detector, image_path: Path, imgsz: int,
                  conf: float, iou: float) -> tuple[list[dict], float, tuple[int, int]]:
    """Run detection. Returns (list of {cls, conf, bbox}, ms, (w, h))."""
    t0 = time.perf_counter()
    res = detector(str(image_path), imgsz=imgsz, conf=conf, iou=iou, verbose=False)
    ms = (time.perf_counter() - t0) * 1000

    detections: list[dict] = []
    img_w, img_h = 0, 0
    if res:
        r = res[0]
        if hasattr(r, "orig_shape"):
            img_h, img_w = r.orig_shape[:2]
        det_names = list(detector.names.values())
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls_idx = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            for (x0, y0, x1, y1), ci, c in zip(xyxy, cls_idx, confs):
                detections.append({
                    "cls":  det_names[ci] if 0 <= ci < len(det_names) else f"cls_{ci}",
                    "conf": float(c),
                    "bbox": (float(x0), float(y0), float(x1), float(y1)),
                })
    return detections, ms, (img_w, img_h)


def main() -> int:
    args = parse_args()

    if args.backend == "hailo":
        print("--backend hailo not implemented yet (Phase 5).", file=sys.stderr)
        print("Use --backend cpu for now.", file=sys.stderr)
        return 2

    if not args.images.is_dir():
        print(f"images dir not found: {args.images}", file=sys.stderr)
        return 1
    if not args.detector.exists():
        print(f"detector not found: {args.detector}", file=sys.stderr)
        return 1

    from ultralytics import YOLO

    logger.info("[load] detector: %s", args.detector)
    detector = YOLO(str(args.detector), task="detect")
    logger.info("[load] detector names: %s", list(detector.names.values()))

    classifier = None
    if args.classifier is not None:
        if not args.classifier.exists():
            print(f"classifier not found: {args.classifier}", file=sys.stderr)
            return 1
        logger.info("[load] classifier: %s", args.classifier)
        classifier = YOLO(str(args.classifier), task="classify")
        logger.info("[load] classifier names: %s", list(classifier.names.values()))
    else:
        logger.info("[load] no classifier — detection-only mode")

    images = _list_images(args.images)
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        print(f"no images under {args.images}", file=sys.stderr)
        return 1
    logger.info("[run] %d images", len(images))

    args.out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_path", "image_w", "image_h",
        "n_total", "n_ripe", "n_unripe", "n_peduncle",
        "diseases_present", "healthy_count",
        "detect_ms", "classify_ms", "total_ms",
    ]
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for i, img_path in enumerate(images):
            t_total = time.perf_counter()
            detections, detect_ms, (img_w, img_h) = _run_detector(
                detector, img_path, args.detect_imgsz, args.conf, args.iou,
            )

            n_ripe = sum(1 for d in detections if d["cls"] == RIPE_CLASS)
            n_unripe = sum(1 for d in detections if d["cls"] == UNRIPE_CLASS)
            n_peduncle = sum(1 for d in detections if d["cls"] == PEDUNCLE_CLASS)
            n_total = len(detections)

            per_det_class, classify_ms = _crop_and_classify(
                img_path, detections, classifier, args.classify_imgsz,
            )

            disease_set = sorted({
                cls for cls in per_det_class
                if cls is not None and cls != HEALTHY_CLS
            })
            healthy_count = sum(1 for cls in per_det_class if cls == HEALTHY_CLS)

            total_ms = (time.perf_counter() - t_total) * 1000
            try:
                rel = img_path.relative_to(REPO_ROOT)
            except ValueError:
                rel = img_path

            writer.writerow({
                "image_path":       str(rel),
                "image_w":          img_w,
                "image_h":          img_h,
                "n_total":          n_total,
                "n_ripe":           n_ripe,
                "n_unripe":         n_unripe,
                "n_peduncle":       n_peduncle,
                "diseases_present": ";".join(disease_set),
                "healthy_count":    healthy_count,
                "detect_ms":        f"{detect_ms:.1f}",
                "classify_ms":      f"{classify_ms:.1f}",
                "total_ms":         f"{total_ms:.1f}",
            })
            if (i + 1) % 50 == 0 or (i + 1) == len(images):
                logger.info(
                    "[progress] %d/%d  last_total=%.1f ms",
                    i + 1, len(images), total_ms,
                )

    logger.info("[done] %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

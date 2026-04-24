"""Per-dataset scanner functions.

Each scan_* function returns a list of ManifestEntry records sorted by
original_path. Walks are deterministic: every Path.rglob/iterdir result is
wrapped in sorted() so two runs over the same on-disk state produce identical
output.

Error handling: a missing or malformed label file logs a warning and the
entry is emitted with source_class_names=[] / project_schema_class=None.
The build never aborts on a single bad image.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Optional

import yaml
from PIL import Image

from src.manifest.hashing import sha256_file
from src.manifest.schema import (
    DETECT_CLASSES,
    DISEASE_CLASSES,
    SOURCE_LICENSES,
    ManifestEntry,
    Split,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Class-name canonicalization
# ---------------------------------------------------------------------------

# Per-source: raw source class name -> canonical project_schema_class string.
# Caller is responsible for stripping leading whitespace from OSF dir names
# before lookup (the OSF " angular" dir literally has a leading space).
_CLASS_MAP: dict[str, dict[str, str]] = {
    "zenodo": {
        "ripe": "ripe",
        "unripe": "unripe",
        "peduncle": "peduncle",
    },
    "kaggle_afzaal": {
        "Angular Leafspot": "angular_leafspot",
        "Anthracnose Fruit Rot": "anthracnose_fruit_rot",
        "Blossom Blight": "blossom_blight",
        "Gray Mold": "gray_mold",
        "Leaf Spot": "leaf_spot",
        "Powdery Mildew Fruit": "powdery_mildew_fruit",
        "Powdery Mildew Leaf": "powdery_mildew_leaf",
    },
    "osf_ej5qv": {
        "angular": "angular_leafspot",
        "anthracnose": "anthracnose_fruit_rot",
        "blossomblight": "blossom_blight",
        "graymold": "gray_mold",
        "leafspot": "leaf_spot",
        "powderyleaf": "powdery_mildew_leaf",
        "powderymildew": "powdery_mildew_fruit",
        "ripe": "ripe",
        "unripe": "unripe",
    },
    "roboflow_afzaal_bbox_v4": {
        "Angular Leafspot": "angular_leafspot",
        "Anthracnose Fruit Rot": "anthracnose_fruit_rot",
        "Blossom Blight": "blossom_blight",
        "Gray Mold": "gray_mold",
        "Leaf Spot": "leaf_spot",
        "Powdery Mildew Fruit": "powdery_mildew_fruit",
        "Powdery Mildew Leaf": "powdery_mildew_leaf",
    },
    "roboflow_matt_lucky": {
        "ripe": "ripe",
        "unripe": "unripe",
    },
    "roboflow_research_proj": {
        "Angular Leafspot": "angular_leafspot",
        "Anthracnose Fruit Rot": "anthracnose_fruit_rot",
        "Blossom Blight": "blossom_blight",
        "Gray Mold": "gray_mold",
        "Healthy Flower": "healthy_flower",
        "Healthy Fruit": "healthy_fruit",
        "Healthy Leaf": "healthy_leaf",
        "Leaf Spot": "leaf_spot",
        "Powdery Mildew Fruit": "powdery_mildew_fruit",
        "Powdery Mildew Leaf": "powdery_mildew_leaf",
    },
    "strawdi": {
        # All StrawDI images map to fruit_unknown regardless of input.
    },
}

_CANONICAL_UNIVERSE = DETECT_CLASSES | DISEASE_CLASSES | {"unknown"}


def _canonicalize(source: str, source_class_name: str) -> str:
    """Map a raw source class name to a canonical project_schema_class string.

    Returns 'unknown' (and logs a warning) if no mapping exists. Never raises.
    StrawDI always returns 'fruit_unknown'.
    """
    if source == "strawdi":
        return "fruit_unknown"
    table = _CLASS_MAP.get(source, {})
    canonical = table.get(source_class_name)
    if canonical is None:
        logger.warning(
            "unrecognized source class: source=%s name=%r — mapping to 'unknown'",
            source,
            source_class_name,
        )
        return "unknown"
    if canonical not in _CANONICAL_UNIVERSE:
        logger.warning(
            "canonical class %r not in DETECT/DISEASE universe (source=%s, src_name=%r)",
            canonical,
            source,
            source_class_name,
        )
    return canonical


def _project_schema_class(canonical_set: set[str]) -> Optional[str]:
    """Collapse a set of canonical labels for one image into a single class
    or None when the image carries multiple distinct canonical classes.
    """
    if len(canonical_set) == 1:
        return next(iter(canonical_set))
    return None


def _rel_path(p: Path) -> str:
    """Return p relative to the repo root as a forward-slash POSIX string.

    Falls back to the raw absolute path if p is outside the repo (should not
    happen with our scanners).
    """
    abs_p = p.resolve() if not p.is_absolute() else p
    try:
        rel = abs_p.relative_to(REPO_ROOT)
    except ValueError:
        return str(abs_p)
    return rel.as_posix()


def _image_size(path: Path) -> tuple[int, int]:
    """Return (width, height) in px. Returns (0, 0) on read failure."""
    try:
        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to read image dims %s: %s", path, exc)
        return 0, 0


# ---------------------------------------------------------------------------
# Zenodo: YOLO bbox; training/ -> train, validation/ -> val
# ---------------------------------------------------------------------------

def _read_yolo_names(yaml_path: Path, names_txt: Path) -> list[str]:
    """Resolve a YOLO names list from a data.yaml / strawberries.yaml or a
    names.txt fallback. Returns [] if neither is parseable.
    """
    if yaml_path.exists():
        try:
            with open(yaml_path, "r", encoding="utf-8") as fh:
                doc = yaml.safe_load(fh)
            names = doc.get("names")
            if isinstance(names, list):
                return [str(n) for n in names]
            if isinstance(names, dict):
                return [str(names[k]) for k in sorted(names)]
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to parse %s: %s", yaml_path, exc)
    if names_txt.exists():
        try:
            return [
                line.strip()
                for line in names_txt.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to read %s: %s", names_txt, exc)
    return []


def _parse_yolo_label(txt_path: Path, class_names: list[str]) -> list[str]:
    """Parse a YOLO bbox .txt file and return raw source class names (one per
    line, deduped+sorted). Tolerates malformed lines.
    """
    if not txt_path.exists():
        return []
    raw_classes: set[str] = set()
    try:
        for line in txt_path.read_text(encoding="utf-8", errors="replace").splitlines():
            tokens = line.strip().split()
            if not tokens:
                continue
            try:
                idx = int(tokens[0])
            except ValueError:
                continue
            if 0 <= idx < len(class_names):
                raw_classes.add(class_names[idx])
            else:
                logger.warning("class idx %d out of range in %s", idx, txt_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to parse YOLO label %s: %s", txt_path, exc)
        return []
    return sorted(raw_classes)


def scan_zenodo(root: Path = Path("data/zenodo")) -> list[ManifestEntry]:
    """Walk Zenodo strawberries detection dataset.

    Layout: <root>/strawberries/{training,validation}/*.{jpg,txt}
    Class list from <root>/strawberries/strawberries.yaml or names.txt.
    """
    root = Path(root)
    sb = root / "strawberries"
    yaml_path = sb / "strawberries.yaml"
    names_txt = sb / "names.txt"
    class_names = _read_yolo_names(yaml_path, names_txt)
    if not class_names:
        logger.warning("zenodo: empty class list (yaml/names.txt missing or unparseable)")

    entries: list[ManifestEntry] = []
    split_dirs: list[tuple[str, Split]] = [
        ("training", "train"),
        ("validation", "val"),
    ]
    for dir_name, split in split_dirs:
        d = sb / dir_name
        if not d.is_dir():
            continue
        for img_path in sorted(d.rglob("*.jpg")):
            txt_path = img_path.with_suffix(".txt")
            raw_names = _parse_yolo_label(txt_path, class_names)
            canon = {_canonicalize("zenodo", n) for n in raw_names}
            w, h = _image_size(img_path)
            entries.append(ManifestEntry(
                sha256=sha256_file(img_path),
                source="zenodo",
                original_path=_rel_path(img_path),
                filename=img_path.name,
                license=SOURCE_LICENSES["zenodo"],
                annotation_type="bbox_yolo",
                split=split,
                project_schema_class=_project_schema_class(canon),
                source_class_names=raw_names,
                width=w,
                height=h,
            ))
    entries.sort(key=lambda e: e["original_path"])
    return entries


# ---------------------------------------------------------------------------
# Kaggle Afzaal disease: LabelMe polygon JSON sibling per image
# ---------------------------------------------------------------------------

def _parse_labelme(json_path: Path) -> list[str]:
    """Read a LabelMe JSON and return sorted unique shape labels."""
    if not json_path.exists():
        return []
    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to parse LabelMe %s: %s", json_path, exc)
        return []
    raw: set[str] = set()
    for shape in doc.get("shapes", []) or []:
        label = shape.get("label")
        if isinstance(label, str) and label:
            raw.add(label)
    return sorted(raw)


def scan_kaggle_disease(root: Path = Path("data/disease")) -> list[ManifestEntry]:
    """Walk Kaggle Afzaal disease dataset.

    Layout:
      <root>/{train,val,test}/*.{jpg,json}
      <root>/Test Disease Severity Level/Level {1,2}/*.{jpg,json}

    Severity-Level dirs are tagged with split=severity_level_{1,2} so they
    don't get double-counted as another 'test' partition.
    """
    root = Path(root)
    entries: list[ManifestEntry] = []

    plain_splits: list[tuple[str, Split]] = [
        ("train", "train"),
        ("val", "val"),
        ("test", "test"),
    ]
    for dir_name, split in plain_splits:
        d = root / dir_name
        if not d.is_dir():
            continue
        for img_path in sorted(d.rglob("*.jpg")):
            json_path = img_path.with_suffix(".json")
            raw_names = _parse_labelme(json_path)
            canon = {_canonicalize("kaggle_afzaal", n) for n in raw_names}
            w, h = _image_size(img_path)
            entries.append(ManifestEntry(
                sha256=sha256_file(img_path),
                source="kaggle_afzaal",
                original_path=_rel_path(img_path),
                filename=img_path.name,
                license=SOURCE_LICENSES["kaggle_afzaal"],
                annotation_type="polygon_labelme",
                split=split,
                project_schema_class=_project_schema_class(canon),
                source_class_names=raw_names,
                width=w,
                height=h,
            ))

    severity_dir = root / "Test Disease Severity Level"
    severity_splits: list[tuple[str, Split]] = [
        ("Level 1", "severity_level_1"),
        ("Level 2", "severity_level_2"),
    ]
    if severity_dir.is_dir():
        for dir_name, split in severity_splits:
            d = severity_dir / dir_name
            if not d.is_dir():
                continue
            for img_path in sorted(d.rglob("*.jpg")):
                json_path = img_path.with_suffix(".json")
                raw_names = _parse_labelme(json_path)
                canon = {_canonicalize("kaggle_afzaal", n) for n in raw_names}
                w, h = _image_size(img_path)
                entries.append(ManifestEntry(
                    sha256=sha256_file(img_path),
                    source="kaggle_afzaal",
                    original_path=_rel_path(img_path),
                    filename=img_path.name,
                    license=SOURCE_LICENSES["kaggle_afzaal"],
                    annotation_type="polygon_labelme",
                    split=split,
                    project_schema_class=_project_schema_class(canon),
                    source_class_names=raw_names,
                    width=w,
                    height=h,
                ))

    entries.sort(key=lambda e: e["original_path"])
    return entries


# ---------------------------------------------------------------------------
# OSF ej5qv: class-subdir layout (note leading-space " angular" preserved
# in original_path, stripped only for class-name lookup).
# ---------------------------------------------------------------------------

def scan_osf_ej5qv(root: Path = Path("data/osf-ej5qv")) -> list[ManifestEntry]:
    root = Path(root)
    base = root / "extracted" / "dataset"
    entries: list[ManifestEntry] = []
    if not base.is_dir():
        return entries
    # Iterate class subdirs in deterministic order. iterdir() is unordered.
    for class_dir in sorted([p for p in base.iterdir() if p.is_dir()],
                            key=lambda p: p.name):
        # Strip leading whitespace from dir name when mapping; keep original
        # in path.
        raw_class_name = class_dir.name.strip()
        canonical = _canonicalize("osf_ej5qv", raw_class_name)
        for img_path in sorted(class_dir.rglob("*.jpg")):
            w, h = _image_size(img_path)
            entries.append(ManifestEntry(
                sha256=sha256_file(img_path),
                source="osf_ej5qv",
                original_path=_rel_path(img_path),
                filename=img_path.name,
                license=SOURCE_LICENSES["osf_ej5qv"],
                annotation_type="class_subdir",
                split="unknown",
                project_schema_class=canonical if canonical != "unknown" else None,
                source_class_names=[raw_class_name] if raw_class_name else [],
                width=w,
                height=h,
            ))
    entries.sort(key=lambda e: e["original_path"])
    return entries


# ---------------------------------------------------------------------------
# Roboflow YOLO bbox: train/valid/test/{images,labels}/
# ---------------------------------------------------------------------------

def scan_roboflow_yolo(root: Path, source: str) -> list[ManifestEntry]:
    """Walk a Roboflow YOLO export.

    source must be one of:
      - "roboflow_afzaal_bbox_v4"
      - "roboflow_matt_lucky"
      - "roboflow_research_proj"
    """
    if source not in SOURCE_LICENSES:
        raise ValueError(f"unknown source: {source}")
    root = Path(root)
    yaml_path = root / "data.yaml"
    class_names = _read_yolo_names(yaml_path, root / "names.txt")
    if not class_names:
        logger.warning("%s: empty class list (data.yaml not found at %s)", source, yaml_path)

    entries: list[ManifestEntry] = []
    split_dirs: list[tuple[str, Split]] = [
        ("train", "train"),
        ("valid", "val"),
        ("test", "test"),
    ]
    for dir_name, split in split_dirs:
        images_dir = root / dir_name / "images"
        labels_dir = root / dir_name / "labels"
        if not images_dir.is_dir():
            continue
        # Sort all candidate image files (.jpg, .jpeg, .png) in one pass.
        candidates: list[Path] = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            candidates.extend(images_dir.rglob(ext))
        for img_path in sorted(candidates):
            txt_path = labels_dir / (img_path.stem + ".txt")
            raw_names = _parse_yolo_label(txt_path, class_names)
            canon = {_canonicalize(source, n) for n in raw_names}
            w, h = _image_size(img_path)
            entries.append(ManifestEntry(
                sha256=sha256_file(img_path),
                source=source,
                original_path=_rel_path(img_path),
                filename=img_path.name,
                license=SOURCE_LICENSES[source],
                annotation_type="bbox_yolo",
                split=split,
                project_schema_class=_project_schema_class(canon),
                source_class_names=raw_names,
                width=w,
                height=h,
            ))
    entries.sort(key=lambda e: e["original_path"])
    return entries


# ---------------------------------------------------------------------------
# StrawDI_Db1: instance PNG masks (label sibling not a manifest entry).
# Only files under <split>/img/ are recorded.
# ---------------------------------------------------------------------------

def scan_strawdi(root: Path = Path("data/strawdi")) -> list[ManifestEntry]:
    root = Path(root)
    base = root / "extracted" / "StrawDI_Db1"
    entries: list[ManifestEntry] = []
    if not base.is_dir():
        return entries
    split_dirs: list[tuple[str, Split]] = [
        ("train", "train"),
        ("val", "val"),
        ("test", "test"),
    ]
    for dir_name, split in split_dirs:
        img_dir = base / dir_name / "img"
        if not img_dir.is_dir():
            continue
        # StrawDI images are .png. Be permissive and accept .jpg too.
        candidates: list[Path] = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            candidates.extend(img_dir.rglob(ext))
        for img_path in sorted(candidates):
            w, h = _image_size(img_path)
            entries.append(ManifestEntry(
                sha256=sha256_file(img_path),
                source="strawdi",
                original_path=_rel_path(img_path),
                filename=img_path.name,
                license=SOURCE_LICENSES["strawdi"],
                annotation_type="mask_png",
                split=split,
                project_schema_class="fruit_unknown",
                source_class_names=["strawberry"],
                width=w,
                height=h,
            ))
    entries.sort(key=lambda e: e["original_path"])
    return entries

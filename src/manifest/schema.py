"""Canonical project schema: license map, class maps, ManifestEntry shape."""
from __future__ import annotations
from typing import TypedDict, Literal, Optional

# Canonical source keys used in MANIFEST entries.
SOURCE_LICENSES = {
    "zenodo":                    "CC-BY-4.0",
    "kaggle_afzaal":             "CC-BY-4.0",
    "osf_ej5qv":                 "CC-BY-4.0",          # disease classes only; ripe/unripe flagged
    "roboflow_afzaal_bbox_v4":   "CC-BY-4.0",
    "roboflow_matt_lucky":       "CC-BY-4.0",
    "roboflow_research_proj":    "MIT",
    "strawdi":                   "non-commercial-academic",
}

# Detector-stage project schema classes.
DETECT_CLASSES = frozenset({
    "ripe", "unripe", "peduncle",
    "fruit_unknown",                  # StrawDI — known-to-be-fruit, ripeness unknown
    "healthy_fruit", "healthy_leaf", "healthy_flower",  # Roboflow research-proj
})

# Disease-stage project schema classes.
DISEASE_CLASSES = frozenset({
    "angular_leafspot",
    "anthracnose_fruit_rot",
    "blossom_blight",
    "gray_mold",
    "leaf_spot",
    "powdery_mildew_fruit",
    "powdery_mildew_leaf",
    "healthy",
})

AnnotationType = Literal["bbox_yolo", "polygon_labelme", "mask_png", "class_subdir"]
Split = Literal["train", "val", "test", "severity_level_1", "severity_level_2", "unknown"]


class ManifestEntry(TypedDict):
    sha256: str                           # lowercase 64-char hex
    source: str                           # key from SOURCE_LICENSES
    original_path: str                    # path relative to repo root, e.g. "data/zenodo/strawberries/training/IMG_0001.jpg"
    filename: str                         # basename of original_path
    license: str                          # value from SOURCE_LICENSES at time of build
    annotation_type: AnnotationType
    split: Split
    project_schema_class: Optional[str]   # canonical class string, None for detection-multi-label images
    source_class_names: list[str]         # raw labels attached to this image in the source dataset (sorted, deduped)
    width: int                            # px
    height: int                           # px

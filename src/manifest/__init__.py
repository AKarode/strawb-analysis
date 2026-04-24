"""Manifest library: per-image provenance, hashing, and dataset scanners.

Every function here is deterministic: same inputs -> byte-identical output.
"""
from src.manifest.schema import (
    ManifestEntry,
    DETECT_CLASSES,
    DISEASE_CLASSES,
    SOURCE_LICENSES,
)
from src.manifest.hashing import sha256_file
from src.manifest.scanners import (
    scan_zenodo,
    scan_kaggle_disease,
    scan_osf_ej5qv,
    scan_roboflow_yolo,
    scan_strawdi,
)

__all__ = [
    "ManifestEntry",
    "DETECT_CLASSES",
    "DISEASE_CLASSES",
    "SOURCE_LICENSES",
    "sha256_file",
    "scan_zenodo",
    "scan_kaggle_disease",
    "scan_osf_ej5qv",
    "scan_roboflow_yolo",
    "scan_strawdi",
]

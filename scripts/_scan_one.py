#!/usr/bin/env python3
"""Worker: scan one dataset and emit its ManifestEntry list as JSON to stdout.

Spawned by scripts/build_manifest.py to isolate each scanner in its own
process. Sidesteps a per-process I/O slowdown observed when running multiple
scanners sequentially in the same Python process on macOS.

Usage:
    python scripts/_scan_one.py <scanner_key> [<root>] [<source_label>]

scanner_key in {zenodo, kaggle_disease, osf, roboflow, strawdi}.
For roboflow, root and source_label are required.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.manifest.scanners import (  # noqa: E402
    scan_zenodo,
    scan_kaggle_disease,
    scan_osf_ej5qv,
    scan_roboflow_yolo,
    scan_strawdi,
)


def main(argv: list[str]) -> int:
    if len(argv) < 1:
        print("usage: _scan_one.py <scanner_key> [args...]", file=sys.stderr)
        return 2
    key = argv[0]
    if key == "zenodo":
        entries = scan_zenodo(REPO_ROOT / "data/zenodo")
    elif key == "kaggle_disease":
        entries = scan_kaggle_disease(REPO_ROOT / "data/disease")
    elif key == "osf":
        entries = scan_osf_ej5qv(REPO_ROOT / "data/osf-ej5qv")
    elif key == "strawdi":
        entries = scan_strawdi(REPO_ROOT / "data/strawdi")
    elif key == "roboflow":
        if len(argv) < 3:
            print("roboflow requires <root> <source_label>", file=sys.stderr)
            return 2
        entries = scan_roboflow_yolo(REPO_ROOT / argv[1], argv[2])
    else:
        print(f"unknown scanner key: {key}", file=sys.stderr)
        return 2
    json.dump(entries, sys.stdout, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

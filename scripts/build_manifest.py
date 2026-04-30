#!/usr/bin/env python3
"""Build data/MANIFEST.json by scanning all 5 verified datasets under data/.

Each scanner runs in its own Python subprocess via scripts/_scan_one.py and
writes its ManifestEntry list to a temp JSON. The parent merges, sorts, and
writes the final MANIFEST.json atomically.

Subprocess isolation sidesteps a per-process macOS I/O slowdown observed when
running multiple scanners sequentially inside one Python interpreter; running
each scanner in a fresh process keeps every scan fast and predictable.

Deterministic: same inputs -> byte-identical output, every run.
Idempotent: safe to re-run; overwrites previous MANIFEST.json atomically.

Usage:
    python scripts/build_manifest.py [--out PATH] [--quiet] [--workers N]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Repo-root import shim (scripts are run from repo root in this project).
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.manifest import (  # noqa: E402
    ManifestEntry,
    SOURCE_LICENSES,
)


# Scanner spec: (worker key, extra args). Order is presentation only — final
# entries are globally sorted by (source, original_path) anyway.
_SCANNER_SPECS: list[tuple[str, list[str]]] = [
    ("zenodo",         []),
    ("kaggle_disease", []),
    ("osf",            []),
    ("roboflow",       ["data/roboflow/afzaal-bbox-v4",       "roboflow_afzaal_bbox_v4"]),
    ("roboflow",       ["data/roboflow/matt-lucky-ripeness",  "roboflow_matt_lucky"]),
    ("roboflow",       ["data/roboflow/research-proj-disease", "roboflow_research_proj"]),
    ("strawdi",        []),
]


def _spawn_scanner(key: str, extra_args: list[str]) -> list[ManifestEntry]:
    """Run one scanner in a fresh Python subprocess; return its entries."""
    cmd = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "scripts/_scan_one.py"),
        key,
        *extra_args,
    ]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"scanner failed key={key} args={extra_args!r}: rc={proc.returncode}\n"
            f"stderr: {proc.stderr}"
        )
    if not proc.stdout.strip():
        raise RuntimeError(f"scanner emitted no output: key={key}")
    entries = json.loads(proc.stdout)
    if not isinstance(entries, list):
        raise RuntimeError(f"scanner output is not a list: key={key}")
    return entries


def build_manifest(workers: int = 4) -> list[ManifestEntry]:
    """Scan all sources concurrently in subprocesses, return merged + sorted entries."""
    entries: list[ManifestEntry] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_spawn_scanner, key, args): (key, args)
            for key, args in _SCANNER_SPECS
        }
        for fut in as_completed(futures):
            key, args = futures[fut]
            sub = fut.result()
            logging.info("  %-40s %d entries", f"{key} {' '.join(args)}".strip(), len(sub))
            entries.extend(sub)
    # Deterministic global order: sort by (source, original_path).
    entries.sort(key=lambda e: (e["source"], e["original_path"]))
    return entries


def write_manifest_atomic(entries: list[ManifestEntry], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2, sort_keys=True, separators=(",", ": "))
        fh.write("\n")
    os.replace(tmp, out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data/MANIFEST.json")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--workers", type=int, default=4,
                    help="concurrent scanner subprocesses (default 4)")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    entries = build_manifest(workers=args.workers)
    write_manifest_atomic(entries, args.out)
    logging.info("manifest: %d entries -> %s", len(entries), args.out)
    # Per-source summary so the run output is self-checking:
    from collections import Counter
    c = Counter(e["source"] for e in entries)
    for k in sorted(c):
        logging.info("  %-30s %d", k, c[k])
    # License sanity (assert the constants we used didn't drift).
    for src in sorted({e["source"] for e in entries}):
        if src not in SOURCE_LICENSES:
            logging.warning("source %s missing from SOURCE_LICENSES", src)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

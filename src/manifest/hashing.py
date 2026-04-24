"""Deterministic streaming SHA256 over file bytes. Never reads wall-clock."""
from __future__ import annotations
import hashlib
from pathlib import Path

_CHUNK = 1 << 20  # 1 MiB


def sha256_file(path: Path | str) -> str:
    """Return lowercase hex sha256 of the file at path. Streams; no mmap, no timestamps.

    Deterministic: same file bytes -> same digest, always.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

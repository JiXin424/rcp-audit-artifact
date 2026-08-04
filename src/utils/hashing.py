#!/usr/bin/env python3
"""SHA-256 helpers for provenance."""
from __future__ import annotations
import hashlib
from pathlib import Path


def sha256_file(path: Path | str, chunk_size: int = 65536) -> str:
    """SHA-256 of a file (hex digest). Returns '' if path does not exist."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_string(s: str) -> str:
    """SHA-256 of a string (hex digest)."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

#!/usr/bin/env python3
"""YAML config loader."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(path: Path | str) -> Dict[str, Any]:
    """Load a YAML config and return it as a dict."""
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_defaults(overrides: Dict, defaults: Dict) -> Dict:
    """Shallow merge override values on top of defaults."""
    return {**defaults, **overrides}

"""BT evaluator model package.

Provenance: copied verbatim from /ssd/xkb4/SignDiff/SLRTP2025_eval/back_translation/
on 2026-08-02. License follows SignDiff/SLRTP2025_eval.
"""
from .bt_model import build_model, SignModel
from .back_translate import make_back_translation_model, back_translate

__all__ = ['build_model', 'SignModel', 'make_back_translation_model', 'back_translate']

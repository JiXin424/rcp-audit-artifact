"""BT evaluator model package.

Provenance: Architecture from the publicly released SLRTP2025 evaluation
codebase (SignDiff/SLRTP2025_eval/back_translation/).
License follows SignDiff/SLRTP2025_eval.
"""
from .bt_model import build_model, SignModel
from .back_translate import make_back_translation_model, back_translate

__all__ = ['build_model', 'SignModel', 'make_back_translation_model', 'back_translate']

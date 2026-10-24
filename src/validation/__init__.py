"""
Validation package for validating biosignal algorithms against standard references.
"""

from src.validation.neurokit_validation import run_neurokit_validation

__all__ = ["run_neurokit_validation"]

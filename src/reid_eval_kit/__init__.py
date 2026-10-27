"""Reproducible person re-identification evaluation."""

from .api import ReIDEvaluator
from .types import EvaluationConfig, EvaluationInput, EvaluationResult
from .exceptions import EvaluationError, InputValidationError

__version__ = "0.1.0"
__all__ = [
    "ReIDEvaluator",
    "EvaluationConfig",
    "EvaluationInput",
    "EvaluationResult",
    "EvaluationError",
    "InputValidationError",
]

class EvaluationError(RuntimeError):
    """Raised when an evaluation cannot produce scientifically valid results."""


class InputValidationError(EvaluationError, ValueError):
    """Raised for malformed feature or metadata inputs."""

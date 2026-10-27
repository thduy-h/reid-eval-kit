from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .exceptions import InputValidationError
from .types import EvaluationConfig


def load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
    elif p.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise InputValidationError(
                "YAML requires the 'yaml' extra: pip install 'reid-eval-kit[yaml]'"
            ) from exc
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise InputValidationError(f"invalid YAML: {exc}") from exc
    else:
        raise InputValidationError("configuration must be JSON or YAML")
    if not isinstance(data, dict):
        raise InputValidationError("configuration must be an object")
    effective_config(data)
    return data


def effective_config(data: dict[str, Any]) -> dict[str, Any]:
    if "evaluation" in data or "schema_version" in data:
        unknown = set(data) - {"schema_version", "evaluation"}
        if unknown:
            raise InputValidationError(
                f"unknown configuration sections: {sorted(unknown)}"
            )
        if data.get("schema_version", 1) != 1:
            raise InputValidationError("unsupported configuration schema_version")
        d = data.get("evaluation", {})
    else:
        d = data
    if not isinstance(d, dict):
        raise InputValidationError("evaluation must be an object")
    unknown = set(d) - set(EvaluationConfig.__dataclass_fields__)
    if unknown:
        raise InputValidationError(f"unknown evaluation keys: {sorted(unknown)}")
    try:
        EvaluationConfig(**d)
    except (TypeError, ValueError) as exc:
        raise InputValidationError(f"invalid evaluation configuration: {exc}") from exc
    return dict(d)

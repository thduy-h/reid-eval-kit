from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any
import numpy as np


@dataclass(frozen=True)
class EvaluationConfig:
    protocol: str = "custom"
    distance: str = "cosine"
    normalize_features: bool = False
    max_rank: int = 50
    query_chunk_size: int | None = None
    junk_ids: tuple[Any, ...] = (-1,)
    exclude_same_id_same_camera: bool = True
    invalid_query_policy: str = "exclude"
    top_k_diagnostics: int = 10
    include_per_query: bool = True

    def __post_init__(self) -> None:
        from .exceptions import InputValidationError

        if self.protocol not in {
            "custom",
            "market1501",
            "msmt17",
        } or self.distance not in {"cosine", "euclidean"}:
            raise InputValidationError("unknown protocol or distance")
        for name in ("max_rank", "top_k_diagnostics", "query_chunk_size"):
            value = getattr(self, name)
            if value is None and name == "query_chunk_size":
                continue
            if type(value) is not int or value < (
                0 if name == "top_k_diagnostics" else 1
            ):
                raise InputValidationError(
                    f"{name} must be an integer in its supported range"
                )
        for name in (
            "normalize_features",
            "exclude_same_id_same_camera",
            "include_per_query",
        ):
            if type(getattr(self, name)) is not bool:
                raise InputValidationError(f"{name} must be boolean")
        if self.invalid_query_policy != "exclude":
            raise InputValidationError("invalid_query_policy must be 'exclude'")
        if not isinstance(self.junk_ids, (list, tuple)) or any(
            type(v) not in (str, int) for v in self.junk_ids
        ):
            raise InputValidationError(
                "junk_ids must contain integer or string identifiers"
            )
        if self.protocol != "custom" and (
            tuple(self.junk_ids) != (-1,) or not self.exclude_same_id_same_camera
        ):
            raise InputValidationError(
                "named protocols have fixed rules; use custom for overrides"
            )

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["junk_ids"] = list(self.junk_ids)
        return d


@dataclass
class EvaluationInput:
    query_features: np.ndarray
    gallery_features: np.ndarray
    query_ids: np.ndarray
    gallery_ids: np.ndarray
    query_camids: np.ndarray
    gallery_camids: np.ndarray
    query_sample_ids: np.ndarray | None = None
    gallery_sample_ids: np.ndarray | None = None


@dataclass
class QueryResult:
    query_index: int
    query_sample_id: Any
    query_pid: Any
    query_camid: Any
    valid: bool
    invalid_reason: str | None
    num_gallery_candidates: int
    num_excluded: int
    num_positives: int
    first_positive_rank: int | None
    last_positive_rank: int | None
    average_precision: float | None
    inverse_negative_penalty: float | None
    rank1_success: int | None
    rank5_success: int | None
    rank10_success: int | None
    top_matches: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class EvaluationResult:
    cmc: list[float | None]
    rank1: float | None
    rank5: float | None
    rank10: float | None
    mAP: float
    mINP: float
    num_queries: int
    num_valid_queries: int
    num_invalid_queries: int
    num_gallery: int
    per_query: list[QueryResult]
    protocol: str
    config: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "tool": "reid-eval-kit",
            "tool_version": "0.1.0",
            "metrics": {
                "cmc": self.cmc,
                "rank1": self.rank1,
                "rank5": self.rank5,
                "rank10": self.rank10,
                "mAP": self.mAP,
                "mINP": self.mINP,
            },
            "counts": {
                "num_queries": self.num_queries,
                "num_valid_queries": self.num_valid_queries,
                "num_invalid_queries": self.num_invalid_queries,
                "num_gallery": self.num_gallery,
            },
            "protocol": self.protocol,
            "evaluation": self.config,
            "metadata": self.metadata,
            "per_query": [asdict(q) for q in self.per_query],
        }

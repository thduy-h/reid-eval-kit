from __future__ import annotations

from typing import Any

import numpy as np

from .exceptions import InputValidationError


def validate_features(x: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim != 2 or 0 in a.shape:
        raise InputValidationError(f"{name} must have shape (N, D) with N,D > 0")
    if a.dtype.kind not in "fiu":
        raise InputValidationError(f"{name} must be a real numeric array")
    a = a.astype(np.float64, copy=False)
    if not np.isfinite(a).all():
        raise InputValidationError(f"{name} contains NaN or infinity")
    return a


def validate_vector(x: Any, n: int, name: str) -> np.ndarray:
    # Inspect sequences before NumPy can silently stringify mixed identifiers.
    if not isinstance(x, np.ndarray):
        values = list(x) if x is not None else []
        kinds = {
            "integer"
            if isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))
            else "string"
            if isinstance(v, (str, np.str_))
            else "unsupported"
            for v in values
        }
        if len(kinds) != 1 or "unsupported" in kinds:
            raise InputValidationError(
                f"{name} must contain consistently typed integers or strings"
            )
    a = np.asarray(x)
    if a.ndim != 1 or len(a) != n:
        raise InputValidationError(f"{name} must be one-dimensional with length {n}")
    if a.dtype.kind not in "iuU":
        raise InputValidationError(f"{name} must contain integers or Unicode strings")
    if a.dtype.kind == "U" and np.any(a == ""):
        raise InputValidationError(f"{name} contains empty identifiers")
    return a


def normalize_rows(a: np.ndarray) -> np.ndarray:
    # Scaling first avoids both underflow for subnormals and overflow for huge
    # finite vectors. Only exactly zero vectors are rejected; no norm threshold.
    scale = np.max(np.abs(a), axis=1)
    if np.any(scale == 0):
        raise InputValidationError("cannot normalize zero-norm feature vectors")
    scaled = a / scale[:, None]
    return scaled / np.sqrt(np.einsum("ij,ij->i", scaled, scaled))[:, None]


def distances(
    q: np.ndarray, g: np.ndarray, kind: str, normalize: bool = False
) -> np.ndarray:
    if kind not in {"cosine", "euclidean"}:
        raise InputValidationError("distance must be 'cosine' or 'euclidean'")
    q = validate_features(q, "query_features")
    g = validate_features(g, "gallery_features")
    if q.shape[1] != g.shape[1]:
        raise InputValidationError("query and gallery feature dimensions differ")
    if normalize or kind == "cosine":
        q, g = normalize_rows(q), normalize_rows(g)
    # A fixed contraction order avoids changing dot-product reductions when the
    # query chunk changes GEMM/GEMV kernels. No approximate tie merging is used.
    if kind == "cosine":
        d = 1.0 - np.clip(np.einsum("id,jd->ij", q, g, optimize=False), -1.0, 1.0)
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            q2 = np.einsum("ij,ij->i", q, q)
            g2 = np.einsum("ij,ij->i", g, g)
            scale = q2[:, None] + g2[None, :]
            d2 = scale - 2 * np.einsum("id,jd->ij", q, g, optimize=False)
            # Relative cancellation near zero or overflow requires direct
            # subtraction. hypot.reduce computes norms without squaring overflow.
            suspect = (
                (d2 <= 64 * np.finfo(float).eps * scale)
                | ~np.isfinite(d2)
                | (scale < np.finfo(float).tiny)
            )
            d = np.sqrt(np.maximum(d2, 0))
            for i in range(len(q)):
                js = np.flatnonzero(suspect[i])
                for start in range(0, len(js), 1024):
                    selected = js[start : start + 1024]
                    d[i, selected] = np.hypot.reduce(q[i] - g[selected], axis=1)
    if not np.isfinite(d).all():
        raise InputValidationError(
            "computed distances overflow float64; rescale features"
        )
    return d


def stable_order(d: np.ndarray) -> np.ndarray:
    if d.ndim != 1 or not np.isfinite(d).all():
        raise InputValidationError(
            "ranking requires a finite one-dimensional distance row"
        )
    return np.argsort(d, kind="stable")

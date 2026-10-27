from __future__ import annotations

import csv
import io
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .exceptions import InputValidationError
from .metrics import validate_features, validate_vector
from .types import EvaluationResult


def load_archive(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    p = Path(path)
    if p.suffix.lower() != ".npz":
        raise InputValidationError(f"archive must be .npz: {p}")
    try:
        loaded = np.load(p, allow_pickle=False)
        if not isinstance(loaded, np.lib.npyio.NpzFile):
            raise InputValidationError(f"{p}: expected an NPZ archive")
        with loaded as z:
            for key in ("features", "pids", "camids"):
                if key not in z:
                    raise InputValidationError(f"{p}: missing key {key}")
            f, pids, cams = z["features"], z["pids"], z["camids"]
            sid = z["sample_ids"] if "sample_ids" in z else None
        validate_features(f, "features")
        validate_vector(pids, len(f), "pids")
        validate_vector(cams, len(f), "camids")
        if sid is not None:
            validate_vector(sid, len(f), "sample_ids")
            if len(np.unique(sid)) != len(sid):
                raise InputValidationError(f"{p}: duplicate sample_ids")
        return f, pids, cams, sid
    except (ValueError, OSError) as exc:
        raise InputValidationError(f"{p}: {exc}") from exc


def _manifest(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    rows: list[dict[str, Any]] = []
    if p.suffix.lower() == ".csv":
        with p.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or len(set(reader.fieldnames)) != len(
                reader.fieldnames
            ):
                raise InputValidationError(f"{p}: missing or duplicate CSV headers")
            rows = list(reader)
    elif p.suffix.lower() in {".jsonl", ".ndjson"}:
        with p.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError as exc:
                    raise InputValidationError(
                        f"{p} row {line_no}: invalid JSON"
                    ) from exc
    else:
        raise InputValidationError("manifest must be CSV or JSONL")
    required = {"sample_id", "pid", "camid", "embedding_path"}
    for i, r in enumerate(rows, 1):
        if not isinstance(r, dict) or not required <= r.keys():
            raise InputValidationError(
                f"{p} row {i}: expected object with fields {sorted(required)}"
            )
        if None in r or any(r[k] is None or r[k] == "" for k in required):
            raise InputValidationError(
                f"{p} row {i}: missing field value or malformed CSV row"
            )
    return rows


def load_manifest(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p = Path(path)
    rows = _manifest(p)
    if not rows:
        raise InputValidationError(f"{p}: empty manifest")
    feats = []
    pids: list[int | str] = []
    cams: list[int | str] = []
    ids: list[int | str] = []
    root = p.parent.resolve()
    dimension = None
    seen = set()
    for i, r in enumerate(rows, 1):
        sid = r["sample_id"]
        if type(sid) not in (int, str):
            raise InputValidationError(
                f"{p} row {i}: sample_id must be integer or string"
            )
        if sid in seen:
            raise InputValidationError(f"{p} row {i}: duplicate sample_id {sid}")
        seen.add(sid)
        raw_path = r["embedding_path"]
        if (
            not isinstance(raw_path, str)
            or "://" in raw_path
            or Path(raw_path).is_absolute()
        ):
            raise InputValidationError(
                f"{p} row {i}: embedding_path must be local and relative"
            )
        ep = (root / raw_path).resolve()
        if not ep.is_relative_to(root):
            raise InputValidationError(
                f"{p} row {i}: embedding_path escapes manifest directory"
            )
        if not ep.is_file() or ep.suffix.lower() != ".npy":
            raise InputValidationError(
                f"{p} row {i}: missing .npy embedding {raw_path}"
            )
        try:
            a = np.load(ep, allow_pickle=False)
            if not isinstance(a, np.ndarray):
                a.close()
                raise ValueError("expected a single NPY array")
            if a.ndim != 1 or a.size == 0:
                raise ValueError("embedding must be a nonempty one-dimensional vector")
            validate_features(a[None, :], "embedding")
            if dimension is not None and len(a) != dimension:
                raise ValueError(f"dimension {len(a)} differs from {dimension}")
        except (ValueError, OSError) as exc:
            raise InputValidationError(f"{p} row {i}, embedding_path: {exc}") from exc
        dimension = len(a)
        feats.append(a)
        ids.append(sid)
        for key, target in (("pid", pids), ("camid", cams)):
            v = r[key]
            if p.suffix.lower() == ".csv" and re.fullmatch(r"-?(?:0|[1-9][0-9]*)", v):
                v = int(v)
            if type(v) not in (int, str):
                raise InputValidationError(
                    f"{p} row {i}, {key}: expected integer or string"
                )
            if target and type(target[0]) is not type(v):
                raise InputValidationError(
                    f"{p} row {i}, {key}: mixed identifier types"
                )
            target.append(v)
    for key, values in (("sample_id", ids), ("pid", pids), ("camid", cams)):
        validate_vector(values, len(rows), key)
    return np.stack(feats), np.asarray(pids), np.asarray(cams), np.asarray(ids)


def _json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def write_result(
    result: EvaluationResult, out: str | Path, overwrite: bool = False
) -> Path:
    d = Path(out)
    # Serialize everything before creating any files, so malformed values cannot
    # leave a plausible-looking partial metric report.
    files = {
        "metrics.json": _json(result.to_dict()),
        "config.json": _json(result.config),
        "run_metadata.json": _json(result.metadata),
    }
    if result.config["include_per_query"]:
        buf = io.StringIO(newline="")
        fields = (
            list(result.per_query[0].__dataclass_fields__) if result.per_query else []
        )
        fields = [x for x in fields if x != "top_matches"]
        writer = csv.DictWriter(buf, fieldnames=fields)
        writer.writeheader()
        for q in result.per_query:
            writer.writerow({k: getattr(q, k) for k in fields})
        files["per_query.csv"] = buf.getvalue()
    buf = io.StringIO(newline="")
    summary = csv.writer(buf)
    summary.writerow(["metric", "value"])
    for name in ("rank1", "rank5", "rank10", "mAP", "mINP"):
        summary.writerow([name, getattr(result, name)])
    files["summary.csv"] = buf.getvalue()
    d.mkdir(parents=True, exist_ok=True)
    # Reject stale per-query files as well when per-query export is disabled.
    for name in {*files, "per_query.csv"}:
        target = d / name
        if target.is_symlink() or (target.exists() and not overwrite):
            raise FileExistsError(f"refusing to overwrite {target}")
    with tempfile.TemporaryDirectory(prefix=".reideval-", dir=d) as stage:
        for name, content in files.items():
            temp = Path(stage) / name
            temp.write_text(content, encoding="utf-8")
            if overwrite:
                os.replace(temp, d / name)
            else:
                os.link(temp, d / name)  # atomic exclusive creation
    if overwrite and not result.config["include_per_query"]:
        (d / "per_query.csv").unlink(missing_ok=True)
    return d


def read_result(path: str | Path) -> dict[str, Any]:
    def reject_constant(token: str) -> None:
        raise InputValidationError(f"nonstandard JSON number: {token}")

    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f, parse_constant=reject_constant)
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise InputValidationError("unsupported or missing report schema_version")
    for key in ("metrics", "counts", "metadata", "evaluation"):
        if not isinstance(data.get(key), dict):
            raise InputValidationError(f"report missing object {key}")
    counts = data["counts"]
    for key in (
        "num_queries",
        "num_gallery",
        "num_valid_queries",
        "num_invalid_queries",
    ):
        if type(counts.get(key)) is not int or counts[key] < 0:
            raise InputValidationError(f"report has invalid count {key}")
    if (
        not counts["num_gallery"]
        or not counts["num_valid_queries"]
        or counts["num_valid_queries"] + counts["num_invalid_queries"]
        != counts["num_queries"]
    ):
        raise InputValidationError("report has inconsistent query/gallery counts")
    cmc = data["metrics"].get("cmc")
    if not isinstance(cmc, list) or not cmc:
        raise InputValidationError("report missing CMC array")
    previous = 0.0
    for rank, value in enumerate(cmc, 1):
        if rank > counts["num_gallery"]:
            if value is not None:
                raise InputValidationError("report has CMC beyond gallery capacity")
        elif type(value) not in (int, float) or not previous <= value <= 1:
            raise InputValidationError(
                "report has nonfinite, out-of-range, or nonmonotone CMC"
            )
        else:
            previous = value
    for key in ("rank1", "rank5", "rank10", "mAP", "mINP"):
        if key not in data["metrics"]:
            raise InputValidationError(f"report missing metric {key}")
        v = data["metrics"][key]
        if v is None and key.startswith("rank"):
            continue
        if type(v) not in (int, float) or not 0 <= v <= 1:
            raise InputValidationError(f"report has invalid metric {key}")
    return data

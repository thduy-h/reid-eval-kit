from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from .api import ReIDEvaluator
from .io import load_archive, load_manifest, write_result, read_result
from .types import EvaluationInput
from .exceptions import EvaluationError
from .config import load_config, effective_config
from .analysis import compatibility_reasons


def _arr(path):
    if Path(path).suffix.lower() != ".npy":
        raise EvaluationError(f"expected NPY file: {path}")
    array = np.load(path, allow_pickle=False)
    if not isinstance(array, np.ndarray):
        array.close()
        raise EvaluationError(f"expected a single NPY array: {path}")
    return array


def cmd_evaluate(a):
    cfg = effective_config(load_config(a.config)) if a.config else {}
    mapping = {
        "top_k_diagnostics": "top_k",
        "include_per_query": "no_per_query",
        "exclude_same_id_same_camera": "no_same_camera",
        "junk_ids": "junk_id",
    }
    aliases = {
        "top_k": "top_k_diagnostics",
        "no_per_query": "include_per_query",
        "no_same_camera": "exclude_same_id_same_camera",
        "junk_id": "junk_ids",
        "no_normalize": "normalize_features",
    }
    explicit = set(a._explicit) | {aliases.get(k, k) for k in a._explicit}
    for k, v in cfg.items():
        if k not in explicit:
            setattr(
                a,
                mapping.get(k, k),
                not v
                if k in {"include_per_query", "exclude_same_id_same_camera"}
                else v,
            )
    output_files = {
        (Path(a.output) / name).resolve()
        for name in (
            "metrics.json",
            "config.json",
            "run_metadata.json",
            "per_query.csv",
            "summary.csv",
        )
    }
    for key, value in vars(a).items():
        if (
            key != "output"
            and isinstance(value, str)
            and Path(value).resolve() in output_files
        ):
            raise EvaluationError(f"output would overwrite input {key}")
    if (
        a.query
        and a.gallery
        and a.query.endswith(".npz")
        and a.gallery.endswith(".npz")
    ):
        qf, qi, qc, qs = load_archive(a.query)
        gf, gi, gc, gs = load_archive(a.gallery)
    elif a.query_manifest and a.gallery_manifest:
        qf, qi, qc, qs = load_manifest(a.query_manifest)
        gf, gi, gc, gs = load_manifest(a.gallery_manifest)
    else:
        if not all(
            (
                a.query_features,
                a.gallery_features,
                a.query_ids,
                a.gallery_ids,
                a.query_camids,
                a.gallery_camids,
            )
        ):
            raise EvaluationError(
                "provide query/gallery .npz, manifests, or all six .npy inputs"
            )
        qf, gf, qi, gi, qc, gc = (
            _arr(x)
            for x in (
                a.query_features,
                a.gallery_features,
                a.query_ids,
                a.gallery_ids,
                a.query_camids,
                a.gallery_camids,
            )
        )
        qs = gs = None
    styles = [
        bool(a.query or a.gallery),
        bool(a.query_manifest or a.gallery_manifest),
        bool(
            a.query_features
            or a.gallery_features
            or a.query_ids
            or a.gallery_ids
            or a.query_camids
            or a.gallery_camids
        ),
    ]
    if sum(styles) != 1:
        raise EvaluationError(
            "choose exactly one input style: NPZ, separate NPY, or manifests"
        )
    junk = tuple(
        int(x) if isinstance(x, str) and x.lstrip("-").isdigit() else x
        for x in ([-1] if a.junk_id is None else a.junk_id)
    )
    ev = ReIDEvaluator(
        a.protocol,
        a.distance,
        a.normalize_features,
        a.max_rank,
        a.query_chunk_size,
        junk,
        not a.no_same_camera,
        a.top_k,
        not a.no_per_query,
    )
    r = ev.evaluate(inputs=EvaluationInput(qf, gf, qi, gi, qc, gc, qs, gs))
    write_result(r, a.output, a.overwrite)
    print(
        f"Protocol: {r.protocol}\nDistance: {a.distance}\nQueries: {r.num_queries}\nValid queries: {r.num_valid_queries}\nInvalid queries: {r.num_invalid_queries}\nGallery: {r.num_gallery}\nRank-1: {r.rank1}\nRank-5: {r.rank5}\nRank-10: {r.rank10}\nmAP: {r.mAP}\nmINP: {r.mINP}\nOutput directory: {a.output}"
    )


def cmd_inspect(a):
    x = read_result(a.metrics)
    print(
        json.dumps(
            {
                k: x.get(k)
                for k in ("metrics", "counts", "protocol", "evaluation", "metadata")
            },
            indent=2,
        )
    )


def cmd_compare(a):
    if len(a.metrics) < 2:
        raise EvaluationError("compare requires at least two reports")
    xs = [read_result(p) for p in a.metrics]
    base = xs[0]
    for i, x in enumerate(xs[1:], 1):
        reasons = compatibility_reasons(base, x)
        if reasons:
            raise EvaluationError(
                f"comparison {i} is incompatible: {', '.join(reasons)}"
            )
    for i, x in enumerate(xs[1:], 1):
        print(f"comparison {i}: compatible; delta = candidate - baseline")
        for m in ("rank1", "rank5", "rank10", "mAP", "mINP"):
            before, after = base["metrics"][m], x["metrics"][m]
            delta = (
                f"{after - before:.8f}"
                if before is not None and after is not None
                else "unavailable"
            )
            print(f"{m}: {before} -> {after} delta={delta}")


def main(argv=None):
    raw = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(prog="reideval", allow_abbrev=False)
    sp = p.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("evaluate", allow_abbrev=False)
    e.add_argument("--config")
    e.add_argument("--query")
    e.add_argument("--gallery")
    e.add_argument("--query-manifest")
    e.add_argument("--gallery-manifest")
    for n in (
        "query-features",
        "gallery-features",
        "query-ids",
        "gallery-ids",
        "query-camids",
        "gallery-camids",
    ):
        e.add_argument("--" + n.replace("_", "-"))
    e.add_argument(
        "--protocol", default="custom", choices=["custom", "market1501", "msmt17"]
    )
    e.add_argument("--distance", default="cosine", choices=["cosine", "euclidean"])
    e.add_argument(
        "--normalize",
        dest="normalize_features",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    e.add_argument("--max-rank", type=int, default=50)
    e.add_argument("--query-chunk-size", type=int)
    e.add_argument("--junk-id", action="append", default=None)
    e.add_argument("--no-same-camera", action="store_true")
    e.add_argument("--top-k", type=int, default=10)
    e.add_argument("--no-per-query", action="store_true")
    e.add_argument("--output", default="results")
    e.add_argument("--overwrite", action="store_true")
    e.set_defaults(func=cmd_evaluate)
    i = sp.add_parser("inspect")
    i.add_argument("metrics")
    i.set_defaults(func=cmd_inspect)
    c = sp.add_parser("compare")
    c.add_argument("metrics", nargs="+")
    c.set_defaults(func=cmd_compare)
    a = p.parse_args(raw)
    a._explicit = {
        k[2:].replace("-", "_")
        for k in raw
        if k.startswith("--")
        for k in [k.split("=", 1)[0]]
    }
    if "normalize" in a._explicit:
        a._explicit.add("normalize_features")
    try:
        return a.func(a) or 0
    except (EvaluationError, OSError, ValueError) as ex:
        print(f"error: {ex}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

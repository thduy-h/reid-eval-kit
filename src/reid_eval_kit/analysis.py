from __future__ import annotations
import numpy as np


def compatibility_reasons(baseline: dict, candidate: dict) -> list[str]:
    """Return reasons two serialized runs cannot support a paired delta."""
    reasons = []
    for key in ("schema_version", "tool", "tool_version", "protocol"):
        if not baseline.get(key) or baseline.get(key) != candidate.get(key):
            reasons.append(key)
    # Execution/diagnostic controls do not change the scientific experiment.
    for key in (
        "protocol",
        "distance",
        "normalize_features",
        "max_rank",
        "junk_ids",
        "exclude_same_id_same_camera",
        "invalid_query_policy",
    ):
        a, b = baseline.get("evaluation", {}), candidate.get("evaluation", {})
        if key not in a or key not in b or a[key] != b[key]:
            reasons.append(f"evaluation.{key}")
    for key in (
        "num_queries",
        "num_gallery",
        "num_valid_queries",
        "num_invalid_queries",
    ):
        a, b = baseline.get("counts", {}), candidate.get("counts", {})
        if key not in a or key not in b or a[key] != b[key]:
            reasons.append(f"counts.{key}")
    bm, cm = baseline.get("metadata", {}), candidate.get("metadata", {})
    for key, label in (
        ("query_fingerprint", "query metadata"),
        ("gallery_fingerprint", "gallery metadata"),
        ("valid_query_fingerprint", "valid query population"),
        ("protocol_version", "protocol version"),
        ("protocol_rules", "protocol rules"),
        ("metric_definitions", "metric definitions"),
        ("ranking", "ranking policy"),
        ("fingerprint_version", "fingerprint version"),
    ):
        if key not in bm or key not in cm or bm[key] != cm[key]:
            reasons.append(label)

    for key in ("query_sample_ids_provided", "gallery_sample_ids_provided"):
        if bm.get(key) is not True or cm.get(key) is not True:
            reasons.append(f"incomplete population evidence: {key}")
    return reasons


def bootstrap(result, n_bootstrap=1000, confidence_level=0.95, seed=0):
    """Percentile bootstrap over valid query AP/INP observations."""
    valid = [q for q in result.per_query if q.valid]
    if not valid:
        raise ValueError("cannot bootstrap without valid queries")
    if type(n_bootstrap) is not int or n_bootstrap < 1 or not 0 < confidence_level < 1:
        raise ValueError("invalid bootstrap settings")
    rng = np.random.default_rng(seed)
    n = len(valid)
    ap = np.asarray([q.average_precision for q in valid])
    inp = np.asarray([q.inverse_negative_penalty for q in valid])
    aps = np.empty(n_bootstrap)
    inps = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        ix = rng.integers(0, n, n)
        aps[i] = ap[ix].mean()
        inps[i] = inp[ix].mean()
    alpha = (1 - confidence_level) / 2
    return {
        "confidence_level": confidence_level,
        "seed": seed,
        "n_bootstrap": n_bootstrap,
        "mAP": (float(np.quantile(aps, alpha)), float(np.quantile(aps, 1 - alpha))),
        "mINP": (float(np.quantile(inps, alpha)), float(np.quantile(inps, 1 - alpha))),
    }

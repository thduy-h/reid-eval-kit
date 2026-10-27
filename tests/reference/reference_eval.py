"""Deliberately independent loop-based retrieval oracle for tests."""


def reference_evaluate(
    distances,
    query_ids,
    gallery_ids,
    query_cams,
    gallery_cams,
    junk_ids=(-1,),
    exclude_same_id_same_camera=True,
):
    per = []
    for row, qpid, qcam in zip(distances, query_ids, query_cams):
        candidates = []
        for gi, (d, gpid, gcam) in enumerate(zip(row, gallery_ids, gallery_cams)):
            if gpid in junk_ids:
                continue
            if exclude_same_id_same_camera and gpid == qpid and gcam == qcam:
                continue
            candidates.append((float(d), gi, gpid == qpid))
        candidates.sort(key=lambda x: (x[0], x[1]))
        relevance = [int(x[2]) for x in candidates]
        positive_ranks = [i + 1 for i, hit in enumerate(relevance) if hit]
        if not positive_ranks:
            per.append(
                {"valid": False, "candidates": len(candidates), "relevance": relevance}
            )
            continue
        hits = 0
        ap_sum = 0.0
        for rank, hit in enumerate(relevance, 1):
            if hit:
                hits += 1
                ap_sum += hits / rank
        npos = len(positive_ranks)
        per.append(
            {
                "valid": True,
                "candidates": len(candidates),
                "relevance": relevance,
                "ap": ap_sum / npos,
                "inp": npos / positive_ranks[-1],
                "first": positive_ranks[0],
                "last": positive_ranks[-1],
            }
        )
    valid = [x for x in per if x["valid"]]
    if not valid:
        raise ValueError("all queries invalid")
    return {
        "per": per,
        "mAP": sum(x["ap"] for x in valid) / len(valid),
        "mINP": sum(x["inp"] for x in valid) / len(valid),
        "cmc": [
            sum(x["first"] <= rank for x in valid) / len(valid)
            for rank in range(1, len(gallery_ids) + 1)
        ],
    }


def direct_distances(query, gallery, metric):
    """Scalar distance oracle, independent of NumPy contractions."""
    import math

    output = []
    for q in query:
        row = []
        for g in gallery:
            if metric == "euclidean":
                row.append(math.dist(q, g))
            else:
                qn, gn = math.hypot(*q), math.hypot(*g)
                row.append(
                    1
                    - max(
                        -1, min(1, math.fsum((x / qn) * (y / gn) for x, y in zip(q, g)))
                    )
                )
        output.append(row)
    return output

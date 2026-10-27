import numpy as np
from reid_eval_kit import ReIDEvaluator


def fixture():
    q = np.array([[1.0, 0.0], [0.0, 1.0]])
    g = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    return (
        q,
        g,
        np.array([1, 2]),
        np.array([1, 2, 3, 4]),
        np.array([1, 1]),
        np.array([2, 2, 2, 2]),
    )


def test_hand_metrics():
    q, g, qi, gi, qc, gc = fixture()
    r = ReIDEvaluator(
        protocol="custom", distance="cosine", normalize_features=True, max_rank=10
    ).evaluate(q, g, qi, gi, qc, gc)
    assert r.rank1 == 1.0 and r.mAP == 1.0 and r.mINP == 1.0


def test_exclusion_and_invalid():
    q = np.array([[1.0, 0.0], [0.0, 1.0]])
    g = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    r = ReIDEvaluator(protocol="market1501").evaluate(
        q,
        g,
        np.array([1, 9]),
        np.array([1, 1, 2]),
        np.array([1, 1]),
        np.array([1, 2, 2]),
    )
    assert r.num_valid_queries == 1 and r.num_invalid_queries == 1


def test_chunk_equivalence():
    q, g, qi, gi, qc, gc = fixture()
    a = ReIDEvaluator(query_chunk_size=None).evaluate(q, g, qi, gi, qc, gc)
    b = ReIDEvaluator(query_chunk_size=1).evaluate(q, g, qi, gi, qc, gc)
    assert a.to_dict()["metrics"] == b.to_dict()["metrics"]

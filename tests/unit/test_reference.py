import numpy as np
import pytest

from reid_eval_kit import ReIDEvaluator
from reid_eval_kit.exceptions import EvaluationError
from tests.reference.reference_eval import direct_distances, reference_evaluate


@pytest.mark.parametrize(
    "relevance,ap,inp,first,last",
    [
        ([1, 0, 1, 0], 5 / 6, 2 / 3, 1, 3),
        ([0, 1, 0, 1], 1 / 2, 1 / 2, 2, 4),
        ([1, 1, 0, 0], 1, 1, 1, 2),
        ([0, 0, 1, 1], (1 / 3 + 2 / 4) / 2, 1 / 2, 3, 4),
        ([1], 1, 1, 1, 1),
        ([0, 1], 1 / 2, 1 / 2, 2, 2),
        ([1, 1, 1], 1, 1, 1, 3),
    ],
)
def test_hand_calculations(relevance, ap, inp, first, last):
    n = len(relevance)
    result = ReIDEvaluator(distance="euclidean", max_rank=n).evaluate(
        np.array([[0.0]]),
        np.arange(n)[:, None],
        np.array([1]),
        np.array([1 if hit else 2 for hit in relevance]),
        np.array([0]),
        np.ones(n, dtype=int),
    )
    assert result.mAP == pytest.approx(ap, abs=1e-15)
    assert result.mINP == pytest.approx(inp, abs=1e-15)
    assert result.per_query[0].first_positive_rank == first
    assert result.per_query[0].last_positive_rank == last
    assert result.cmc == [float(k >= first) for k in range(1, n + 1)]


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_randomized_against_scalar_oracle(seed, metric, dtype):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(7, 5)).astype(dtype)
    g = rng.normal(size=(21, 5)).astype(dtype)
    qi = np.arange(7)
    gi = np.tile(qi, 3)
    qc = np.zeros(7, dtype=int)
    gc = np.repeat([0, 1, 2], 7)
    oracle = reference_evaluate(
        direct_distances(q.tolist(), g.tolist(), metric), qi, gi, qc, gc
    )
    r = ReIDEvaluator(distance=metric, max_rank=21, query_chunk_size=2).evaluate(
        q, g, qi, gi, qc, gc
    )
    assert r.mAP == pytest.approx(oracle["mAP"], abs=1e-15)
    assert r.mINP == pytest.approx(oracle["mINP"], abs=1e-15)
    for actual, expected in zip(r.per_query, oracle["per"]):
        assert actual.average_precision == pytest.approx(expected["ap"], abs=1e-15)
        assert actual.inverse_negative_penalty == pytest.approx(
            expected["inp"], abs=1e-15
        )
        assert actual.first_positive_rank == expected["first"]
        assert actual.last_positive_rank == expected["last"]
    assert r.cmc[-1] == 1
    assert all(0 <= v <= 1 for v in r.cmc)
    assert r.cmc == sorted(r.cmc)


def test_different_filtered_lengths_and_query_denominator():
    # q0 drops three same-camera positives; q1 drops one; q2 is absent.
    q = np.array([[0.0], [2.0], [9.0]])
    g = np.array([[0.0], [0.1], [0.2], [1.0], [2.0], [3.0]])
    qi = np.array([1, 2, 99])
    gi = np.array([1, 1, 1, 1, 2, 2])
    qc = np.zeros(3, dtype=int)
    gc = np.array([0, 0, 0, 1, 0, 1])
    oracle = reference_evaluate(
        direct_distances(q.tolist(), g.tolist(), "euclidean"), qi, gi, qc, gc
    )
    r = ReIDEvaluator(distance="euclidean", max_rank=6).evaluate(q, g, qi, gi, qc, gc)
    assert [x.num_gallery_candidates for x in r.per_query] == [3, 5, 6]
    assert (r.num_valid_queries, r.num_invalid_queries) == (2, 1)
    assert r.mAP == pytest.approx(oracle["mAP"])
    assert r.mINP == pytest.approx(oracle["mINP"])
    assert r.cmc == oracle["cmc"]
    assert r.per_query[2].rank10_success is None
    assert isinstance(r.per_query[2].top_matches, list)


def test_all_positives_removed_and_junk_only():
    for gpids, gcams in (([1], [0]), ([-1], [1]), ([9], [1])):
        with pytest.raises(EvaluationError, match="all queries"):
            ReIDEvaluator(distance="euclidean").evaluate(
                np.array([[0.0]]),
                np.array([[0.0]]),
                np.array([1]),
                np.array(gpids),
                np.array([0]),
                np.array(gcams),
            )


def test_permutation_and_noncontiguous_renaming_invariance():
    q = np.array([[0.0], [2.0]])
    g = np.array([[0.2], [1.1], [2.3], [4.0]])
    qi = np.array([1, 2])
    gi = np.array([1, 9, 2, 1])
    qc = np.array([0, 0])
    gc = np.array([1, 1, 1, 2])
    e = ReIDEvaluator(distance="euclidean")
    a = e.evaluate(q, g, qi, gi, qc, gc)
    ix = np.array([3, 1, 0, 2])
    b = e.evaluate(q, g[ix], qi, gi[ix], qc, gc[ix])
    c = e.evaluate(q, g, qi * 17 + 4, gi * 17 + 4, qc, gc)
    assert a.to_dict()["metrics"] == b.to_dict()["metrics"] == c.to_dict()["metrics"]

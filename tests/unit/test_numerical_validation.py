import math

import numpy as np
import pytest

from reid_eval_kit import ReIDEvaluator
from reid_eval_kit.exceptions import InputValidationError
from reid_eval_kit.metrics import distances, stable_order


@pytest.mark.parametrize("scale", [1e-300, 1e-150, 1.0, 1e150, 1e300])
@pytest.mark.parametrize("normalize", [False, True])
def test_cosine_norm_extremes(scale, normalize):
    q = np.array([[scale, 0.0]])
    g = np.array([[scale, 0.0], [0.0, scale], [-scale, 0.0]])
    np.testing.assert_allclose(
        distances(q, g, "cosine", normalize), [[0, 1, 2]], atol=0, rtol=0
    )


@pytest.mark.parametrize(
    "offset,delta", [(1e12, 1.0), (1e150, 1e140), (1e300, 1e290), (0.0, 1e-300)]
)
def test_euclidean_cancellation_and_extremes(offset, delta):
    q = np.array([[offset, offset]])
    g = np.array([[offset + delta, offset - delta], [offset, offset]])
    expected = [math.dist(q[0], row) for row in g]
    np.testing.assert_allclose(
        distances(q, g, "euclidean")[0], expected, rtol=3e-16, atol=0
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("metric", ["cosine", "euclidean"])
def test_distance_reference_and_chunk_bits(dtype, metric):
    rng = np.random.default_rng(12)
    q = rng.normal(size=(11, 19)).astype(dtype)
    g = rng.normal(size=(17, 19)).astype(dtype)
    a = distances(q, g, metric)
    for chunk in (1, 2, 7, 40):
        b = np.concatenate(
            [distances(q[i : i + chunk], g, metric) for i in range(0, len(q), chunk)]
        )
        np.testing.assert_array_equal(a, b)
    if metric == "euclidean":
        expected = np.array([[math.dist(x, y) for y in g.tolist()] for x in q.tolist()])
        np.testing.assert_allclose(a, expected, rtol=5e-16, atol=0)


def test_near_ties_are_not_approximately_collapsed():
    d = np.array([1 + 1e-12, 1.0, 1.0])
    assert stable_order(d).tolist() == [1, 2, 0]
    q = np.array([[1.0, 0.0]])
    g = np.array([[1.0, 1e-5], [1.0, 1e-6]])
    assert stable_order(distances(q, g, "euclidean")[0]).tolist() == [1, 0]


def test_normalization_changes_euclidean_and_cosine_is_intrinsic():
    q = np.array([[2.0, 0.0]])
    g = np.array([[1.0, 1.0], [10.0, 0.0]])
    assert stable_order(distances(q, g, "euclidean", False)[0]).tolist() == [0, 1]
    assert stable_order(distances(q, g, "euclidean", True)[0]).tolist() == [1, 0]
    np.testing.assert_array_equal(
        distances(q, g, "cosine", False), distances(q, g, "cosine", True)
    )


@pytest.mark.parametrize(
    "metric,normalize", [("cosine", False), ("cosine", True), ("euclidean", True)]
)
def test_zero_vector_rejection(metric, normalize):
    with pytest.raises(InputValidationError, match="zero-norm"):
        distances(np.zeros((1, 2)), np.ones((1, 2)), metric, normalize)


def test_zero_euclidean_and_overflow_error():
    assert distances(np.zeros((1, 2)), np.zeros((1, 2)), "euclidean")[0, 0] == 0
    with pytest.raises(InputValidationError, match="overflow"):
        distances(np.array([[1e308]]), np.array([[-1e308]]), "euclidean")


@pytest.mark.parametrize(
    "row", [np.array([float("nan")]), np.array([float("inf")]), np.ones((1, 2))]
)
def test_ranking_rejects_nonfinite_or_wrong_shape(row):
    with pytest.raises(InputValidationError):
        stable_order(row)


@pytest.mark.parametrize(
    "value",
    [
        np.array([1.0]),
        np.array([True]),
        np.array([1 + 1j]),
        np.array([object()], dtype=object),
    ],
)
def test_identifier_types_rejected(value):
    with pytest.raises(InputValidationError):
        ReIDEvaluator(distance="euclidean").evaluate(
            np.ones((1, 1)),
            np.ones((1, 1)),
            value,
            np.array([1]),
            np.array([0]),
            np.array([1]),
        )


def test_mixed_identifier_sequence_is_not_stringified():
    with pytest.raises(InputValidationError, match="consistently"):
        ReIDEvaluator(distance="euclidean").evaluate(
            np.ones((2, 1)), np.ones((1, 1)), [1, "a"], [1], [0, 0], [1]
        )

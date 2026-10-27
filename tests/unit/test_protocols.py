import numpy as np
import pytest

from reid_eval_kit import EvaluationConfig, ReIDEvaluator
from reid_eval_kit.exceptions import InputValidationError
from reid_eval_kit.protocols import resolve_protocol


@pytest.mark.parametrize("name", ["market1501", "msmt17", "custom"])
def test_protocol_masks_exclude_only_same_person_and_camera(name):
    p = resolve_protocol(name)
    candidates, positives = p.mask(5, 2, np.array([5, 5, 7, 7]), np.array([2, 3, 2, 3]))
    assert candidates.tolist() == [False, True, True, True]
    assert positives.tolist() == [False, True, False, False]
    assert p.rules()["exclude_same_id_same_camera"] is True


@pytest.mark.parametrize("name", ["market1501", "msmt17"])
def test_named_rules_cannot_be_silently_overridden(name):
    with pytest.raises(InputValidationError, match="fixed rules"):
        ReIDEvaluator(protocol=name, exclude_same_id_same_camera=False)
    with pytest.raises(InputValidationError, match="fixed rules"):
        ReIDEvaluator(protocol=name, junk_ids=(0,))


def test_msmt_zero_is_person_and_minus_one_has_no_upstream_junk_convention():
    p = resolve_protocol("msmt17")
    assert p.junk_ids == ()
    r = ReIDEvaluator(protocol="msmt17", distance="euclidean").evaluate(
        np.array([[0.0]]),
        np.array([[0.0]]),
        np.array([0]),
        np.array([0]),
        np.array([0]),
        np.array([14]),
    )
    assert r.mAP == 1.0 and r.metadata["protocol_rules"]["junk_ids"] == []
    assert "experimental" in r.metadata["protocol_status"]
    with pytest.raises(InputValidationError, match="negative"):
        ReIDEvaluator(protocol="msmt17", distance="euclidean").evaluate(
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.array([0]),
            np.array([-1]),
            np.array([0]),
            np.array([14]),
        )


def test_msmt_joint_offset_labels_preserves_equality():
    data = dict(
        query_features=np.array([[0.0]]),
        gallery_features=np.array([[0.0], [1.0]]),
        query_ids=np.array([0]),
        gallery_ids=np.array([0, 1]),
        query_camids=np.array([0]),
        gallery_camids=np.array([1, 2]),
    )
    e = ReIDEvaluator(protocol="msmt17", distance="euclidean")
    a = e.evaluate(**data)
    data["query_ids"] = data["query_ids"] + 1041
    data["gallery_ids"] = data["gallery_ids"] + 1041
    assert e.evaluate(**data).to_dict()["metrics"] == a.to_dict()["metrics"]


@pytest.mark.parametrize("name,invalid_camera", [("market1501", 6), ("msmt17", 15)])
def test_named_protocol_camera_and_identifier_validation(name, invalid_camera):
    args = dict(
        query_features=np.array([[0.0]]),
        gallery_features=np.array([[0.0]]),
        query_ids=np.array([1]),
        gallery_ids=np.array([1]),
        query_camids=np.array([0]),
        gallery_camids=np.array([invalid_camera]),
    )
    with pytest.raises(InputValidationError, match="zero-based"):
        ReIDEvaluator(protocol=name, distance="euclidean").evaluate(**args)
    args.update(gallery_camids=np.array([1]), query_ids=np.array(["1"]))
    with pytest.raises(InputValidationError):
        ReIDEvaluator(protocol=name, distance="euclidean").evaluate(**args)


def test_named_ranks_do_not_depend_on_cmc_max_rank():
    r = ReIDEvaluator(distance="euclidean", max_rank=1).evaluate(
        np.array([[0.0]]),
        np.arange(10)[:, None],
        np.array([1]),
        np.array([2, 2, 2, 2, 1, 2, 2, 2, 2, 2]),
        np.array([0]),
        np.ones(10, dtype=int),
    )
    assert r.cmc == [0.0] and r.rank1 == 0 and r.rank5 == 1 and r.rank10 == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_rank": 0},
        {"max_rank": True},
        {"query_chunk_size": 0},
        {"query_chunk_size": 1.5},
        {"top_k_diagnostics": -1},
        {"normalize_features": "yes"},
        {"invalid_query_policy": "include"},
        {"junk_ids": [1.5]},
    ],
)
def test_config_validation(kwargs):
    with pytest.raises(InputValidationError):
        EvaluationConfig(**kwargs)


def test_unsafe_signed_unsigned_identity_types_rejected():
    with pytest.raises(InputValidationError, match="signed/unsigned"):
        ReIDEvaluator(distance="euclidean").evaluate(
            np.ones((1, 1)),
            np.ones((1, 1)),
            np.array([2**63 - 1], dtype=np.int64),
            np.array([2**63], dtype=np.uint64),
            np.array([0]),
            np.array([1]),
        )

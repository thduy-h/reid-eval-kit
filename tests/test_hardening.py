import json
import numpy as np
import pytest

from reid_eval_kit import EvaluationConfig, EvaluationInput, ReIDEvaluator
from reid_eval_kit.analysis import bootstrap, compatibility_reasons
from reid_eval_kit.exceptions import EvaluationError, InputValidationError
from reid_eval_kit.io import load_archive, load_manifest, read_result, write_result
from reid_eval_kit.metrics import distances
from reid_eval_kit.protocols import resolve_protocol
from reid_eval_kit.cli import main
from tests.reference.reference_eval import reference_evaluate


def run(q, g, qids, gids, qcams, gcams, **kwargs):
    sample_args = {
        k: kwargs.pop(k)
        for k in ("query_sample_ids", "gallery_sample_ids")
        if k in kwargs
    }
    return ReIDEvaluator(**kwargs).evaluate(
        q, g, qids, gids, qcams, gcams, **sample_args
    )


def test_hand_calculated_ap_and_inp_against_independent_oracle():
    q = np.array([[0.0]])
    g = np.array([[0.0], [1.0], [2.0], [3.0]])
    r = run(
        q,
        g,
        np.array([1]),
        np.array([1, 7, 1, 8]),
        np.array([1]),
        np.array([2, 2, 2, 2]),
        distance="euclidean",
        max_rank=4,
    )
    oracle = reference_evaluate(
        distances(q, g, "euclidean"), [1], [1, 7, 1, 8], [1], [2, 2, 2, 2]
    )
    assert oracle["per"][0]["relevance"] == [1, 0, 1, 0]
    assert r.mAP == pytest.approx(5 / 6) and r.mINP == pytest.approx(2 / 3)
    assert (
        r.per_query[0].first_positive_rank == 1
        and r.per_query[0].last_positive_rank == 3
    )
    assert r.mAP == pytest.approx(oracle["mAP"]) and r.mINP == pytest.approx(
        oracle["mINP"]
    )


def test_single_positive_and_cmc_ranks():
    q = np.array([[0.0]])
    g = np.arange(10, dtype=float)[:, None]
    r = run(
        q,
        g,
        np.array([42]),
        np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 42]),
        np.array([1]),
        np.full(10, 2),
        distance="euclidean",
        max_rank=10,
    )
    assert r.per_query[0].first_positive_rank == 10
    assert r.rank1 == 0.0 and r.rank5 == 0.0 and r.rank10 == 1.0
    assert r.mAP == pytest.approx(0.1) and r.mINP == pytest.approx(0.1)
    assert all(b is None or a <= b for a, b in zip(r.cmc, r.cmc[1:]))


def test_valid_invalid_and_all_invalid_failure():
    q = np.array([[0.0], [1.0]])
    g = np.array([[0.0]])
    r = run(
        q,
        g,
        np.array([1, 2]),
        np.array([1]),
        np.array([1, 1]),
        np.array([2]),
        distance="euclidean",
    )
    assert (r.num_queries, r.num_valid_queries, r.num_invalid_queries) == (2, 1, 1)
    assert r.per_query[1].invalid_reason == "identity_absent"
    with pytest.raises(EvaluationError):
        run(
            q,
            g,
            np.array([7, 8]),
            np.array([1]),
            np.array([1, 1]),
            np.array([2]),
            distance="euclidean",
        )


def test_same_camera_exclusion_and_cross_camera_positive():
    q = np.array([[0.0]])
    g = np.array([[0.0], [0.1], [1.0]])
    r = run(
        q,
        g,
        np.array([1]),
        np.array([1, 1, 9]),
        np.array([3]),
        np.array([3, 4, 3]),
        distance="euclidean",
    )
    assert r.per_query[0].num_excluded == 1 and r.per_query[0].num_positives == 1
    custom = run(
        q,
        g,
        np.array([1]),
        np.array([1, 1, 9]),
        np.array([3]),
        np.array([3, 4, 3]),
        protocol="custom",
        exclude_same_id_same_camera=False,
        distance="euclidean",
    )
    assert custom.per_query[0].num_positives == 2


def test_market_zero_preserved_and_minus_one_junk():
    q = np.array([[0.0]])
    g = np.array([[0.0], [0.1], [0.2], [0.3]])
    r = run(
        q,
        g,
        np.array([0]),
        np.array([0, 0, -1, 4]),
        np.array([1]),
        np.array([2, 1, 2, 2]),
        protocol="market1501",
        distance="euclidean",
    )
    assert r.num_valid_queries == 1 and r.per_query[0].num_positives == 1
    assert r.per_query[0].num_excluded == 2
    assert resolve_protocol("market1501").version.startswith("market1501-")


def test_msmt17_explicit_protocol_and_rule():
    p = resolve_protocol("msmt17")
    assert p.name == "msmt17" and p.version.startswith("msmt17-")
    q = np.array([[0.0]])
    g = np.array([[0.0], [1.0]])
    r = run(
        q,
        g,
        np.array([4]),
        np.array([4, 4]),
        np.array([1]),
        np.array([1, 2]),
        protocol="msmt17",
        distance="euclidean",
    )
    assert r.per_query[0].num_positives == 1 and r.per_query[0].num_excluded == 1


def test_custom_junk_and_exclusion_configuration():
    q = np.array([[0.0]])
    g = np.array([[0.0], [0.1], [0.2]])
    r = run(
        q,
        g,
        np.array([5]),
        np.array([5, 0, 6]),
        np.array([1]),
        np.array([2, 2, 2]),
        protocol="custom",
        junk_ids=(0,),
        distance="euclidean",
    )
    assert r.per_query[0].num_excluded == 1 and r.per_query[0].num_positives == 1


def test_stable_equal_distance_ties_follow_gallery_order():
    q = np.array([[1.0, 0.0]])
    g = np.array([[0.0, 1.0], [0.0, -1.0]])
    a = run(q, g, np.array([7]), np.array([99, 7]), np.array([1]), np.array([2, 2]))
    b = run(
        q, g[::-1], np.array([7]), np.array([7, 99]), np.array([1]), np.array([2, 2])
    )
    assert (
        a.per_query[0].first_positive_rank == 2
        and b.per_query[0].first_positive_rank == 1
    )


def test_distance_numerical_edges_and_precision():
    x = np.array([[1.0, 0.0]], dtype=np.float32)
    y = np.array([[1.0, 0.0]], dtype=np.float32)
    assert distances(x, y, "cosine")[0, 0] == pytest.approx(0.0)
    assert distances(x, -y, "cosine")[0, 0] == pytest.approx(2.0)
    assert distances(x, np.array([[0.0, 1.0]]), "cosine")[0, 0] == pytest.approx(1.0)
    assert distances(np.array([[1e12, 1e12]]), np.array([[1e12, 1e12]]), "euclidean")[
        0, 0
    ] == pytest.approx(0.0)
    with pytest.raises(InputValidationError):
        distances(np.zeros((1, 2)), x, "cosine")
    with pytest.raises(InputValidationError):
        distances(x, y, "manhattan")
    assert (
        distances(x.astype(np.float64), y.astype(np.float64), "euclidean").dtype
        == np.float64
    )


def test_feature_validation_rejects_nan_shape_and_complex():
    good = np.array([[0.0]])
    with pytest.raises(InputValidationError):
        run(
            np.array([[np.nan]]),
            good,
            np.array([1]),
            np.array([1]),
            np.array([1]),
            np.array([2]),
        )
    with pytest.raises(InputValidationError):
        run(
            np.array([0.0]),
            good,
            np.array([1]),
            np.array([1]),
            np.array([1]),
            np.array([2]),
        )
    with pytest.raises(InputValidationError):
        run(
            np.array([[1 + 1j]]),
            good,
            np.array([1]),
            np.array([1]),
            np.array([1]),
            np.array([2]),
        )


def test_full_and_chunked_equivalence():
    rng = np.random.default_rng(5)
    q = rng.normal(size=(7, 4))
    g = rng.normal(size=(11, 4))
    qi = np.arange(7)
    gi = np.arange(11)
    qc = np.arange(7) % 2
    gc = np.arange(11) % 3
    full = run(q, g, qi, gi, qc, gc, query_chunk_size=None, max_rank=11)
    for size in (1, 2, 7, 20):
        chunked = run(q, g, qi, gi, qc, gc, query_chunk_size=size, max_rank=11)
        assert chunked.to_dict()["metrics"] == pytest.approx(full.to_dict()["metrics"])


def test_rank_unavailable_when_gallery_smaller_than_requested():
    q = np.array([[0.0]])
    g = np.array([[0.0], [1.0]])
    r = run(
        q,
        g,
        np.array([1]),
        np.array([1, 2]),
        np.array([1]),
        np.array([2, 2]),
        distance="euclidean",
        max_rank=10,
    )
    assert r.rank1 == 1.0 and r.rank5 is None and r.rank10 is None
    assert r.cmc[:2] == [1.0, 1.0] and all(x is None for x in r.cmc[2:])


def test_typed_input_and_config_object():
    data = EvaluationInput(
        np.array([[0.0]]),
        np.array([[0.0]]),
        np.array([1]),
        np.array([1]),
        np.array([1]),
        np.array([2]),
    )
    r = ReIDEvaluator(
        config=EvaluationConfig(distance="euclidean", max_rank=3)
    ).evaluate(inputs=data)
    assert r.mAP == 1.0 and r.config["distance"] == "euclidean"


def test_npz_csv_jsonl_loading_and_malformed_inputs(tmp_path):
    emb = tmp_path / "q.npy"
    np.save(emb, np.array([1.0, 2.0]))
    np.savez(
        tmp_path / "data.npz",
        features=np.array([[1.0, 2.0]]),
        pids=np.array([1]),
        camids=np.array([2]),
        sample_ids=np.array(["q"]),
    )
    f, p, c, s = load_archive(tmp_path / "data.npz")
    assert f.shape == (1, 2) and s.tolist() == ["q"]
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(
        "sample_id,pid,camid,embedding_path\nq,1,2,q.npy\n", encoding="utf-8"
    )
    assert load_manifest(csv_path)[0].shape == (1, 2)
    jsonl = tmp_path / "data.jsonl"
    jsonl.write_text(
        json.dumps({"sample_id": "q", "pid": 1, "camid": 2, "embedding_path": "q.npy"})
        + "\n",
        encoding="utf-8",
    )
    assert load_manifest(jsonl)[1].tolist() == [1]
    bad = tmp_path / "bad.npz"
    np.savez(bad, features=np.array([[1.0]]))
    with pytest.raises(InputValidationError):
        load_archive(bad)
    dup = tmp_path / "dup.csv"
    dup.write_text(
        "sample_id,pid,camid,embedding_path\nq,1,2,q.npy\nq,1,2,q.npy\n",
        encoding="utf-8",
    )
    with pytest.raises(InputValidationError):
        load_manifest(dup)


def test_reports_json_csv_and_unavailable_values(tmp_path):
    q = np.array([[0.0]])
    g = np.array([[0.0]])
    r = run(
        q,
        g,
        np.array([1]),
        np.array([1]),
        np.array([1]),
        np.array([2]),
        distance="euclidean",
        max_rank=10,
    )
    out = write_result(r, tmp_path)
    payload = read_result(out / "metrics.json")
    assert payload["metrics"]["rank5"] is None
    assert (out / "per_query.csv").is_file() and (out / "summary.csv").is_file()
    with pytest.raises(FileExistsError):
        write_result(r, tmp_path)


def test_comparison_distinguishes_population_from_features():
    q = np.array([[0.0]])
    g = np.array([[0.0]])
    a = run(
        q,
        g,
        np.array([1]),
        np.array([1]),
        np.array([1]),
        np.array([2]),
        distance="euclidean",
        query_sample_ids=np.array(["q"]),
        gallery_sample_ids=np.array(["g"]),
    )
    b = run(
        q + 1,
        g + 1,
        np.array([1]),
        np.array([1]),
        np.array([1]),
        np.array([2]),
        distance="euclidean",
        query_sample_ids=np.array(["q"]),
        gallery_sample_ids=np.array(["g"]),
    )
    assert compatibility_reasons(a.to_dict(), b.to_dict()) == []
    c = run(
        q,
        g,
        np.array([1]),
        np.array([1]),
        np.array([9]),
        np.array([2]),
        distance="euclidean",
        query_sample_ids=np.array(["q"]),
        gallery_sample_ids=np.array(["g"]),
    )
    assert "query metadata" in compatibility_reasons(a.to_dict(), c.to_dict())


def test_bootstrap_is_seeded_and_validates_inputs():
    q = np.array([[0.0], [0.1]])
    g = np.array([[0.0], [1.0]])
    r = run(
        q,
        g,
        np.array([1, 2]),
        np.array([1, 2]),
        np.array([1, 1]),
        np.array([2, 2]),
        distance="euclidean",
    )
    a = bootstrap(r, n_bootstrap=50, seed=7)
    b = bootstrap(r, n_bootstrap=50, seed=7)
    assert a == b and a["seed"] == 7
    assert a["mAP"] == (0.5, 1.0) and a["mINP"] == (0.5, 1.0)
    with pytest.raises(ValueError):
        bootstrap(r, n_bootstrap=0)


def test_cli_api_parity_and_inspect(tmp_path, capsys):
    q = np.array([[0.0], [2.0]])
    g = np.array([[0.0], [1.0], [2.0]])
    qi = np.array([1, 2])
    gi = np.array([1, 9, 2])
    qc = np.array([1, 1])
    gc = np.array([2, 2, 2])
    qpath = tmp_path / "q.npz"
    gpath = tmp_path / "g.npz"
    np.savez(qpath, features=q, pids=qi, camids=qc)
    np.savez(gpath, features=g, pids=gi, camids=gc)
    direct = run(q, g, qi, gi, qc, gc, distance="euclidean")
    out = tmp_path / "results"
    assert (
        main(
            [
                "evaluate",
                "--query",
                str(qpath),
                "--gallery",
                str(gpath),
                "--distance",
                "euclidean",
                "--output",
                str(out),
            ]
        )
        == 0
    )
    payload = read_result(out / "metrics.json")
    assert payload["metrics"]["mAP"] == pytest.approx(direct.mAP)
    assert main(["inspect", str(out / "metrics.json")]) == 0
    assert "Protocol:" in capsys.readouterr().out

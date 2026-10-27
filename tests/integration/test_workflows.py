import copy
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from reid_eval_kit import ReIDEvaluator
from reid_eval_kit.analysis import bootstrap, compatibility_reasons
from reid_eval_kit.cli import main
from reid_eval_kit.config import load_config
from reid_eval_kit.exceptions import InputValidationError
from reid_eval_kit.hashing import array_fingerprint
from reid_eval_kit.io import load_manifest, read_result, write_result


@pytest.fixture
def inputs(tmp_path):
    data = dict(
        query_features=np.array([[1.0, 0.0], [0.0, 1.0]]),
        gallery_features=np.array([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [-1.0, 0.0]]),
        query_ids=np.array([1, 2]),
        gallery_ids=np.array([1, 9, 2, 1]),
        query_camids=np.array([0, 0]),
        gallery_camids=np.array([1, 1, 1, 1]),
    )
    qids = np.array(["q0", "q1"])
    gids = np.array(["g0", "g1", "g2", "g3"])
    for key, value in data.items():
        np.save(tmp_path / f"{key}.npy", value)
    for split, sids in (("query", qids), ("gallery", gids)):
        np.savez(
            tmp_path / f"{split}.npz",
            features=data[f"{split}_features"],
            pids=data[f"{split}_ids"],
            camids=data[f"{split}_camids"],
            sample_ids=sids,
        )
        rows = []
        for i, sid in enumerate(sids):
            np.save(tmp_path / f"{sid}.npy", data[f"{split}_features"][i])
            rows.append(
                dict(
                    sample_id=str(sid),
                    pid=int(data[f"{split}_ids"][i]),
                    camid=int(data[f"{split}_camids"][i]),
                    embedding_path=f"{sid}.npy",
                )
            )
        (tmp_path / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n"
        )
        (tmp_path / f"{split}.csv").write_text(
            "sample_id,pid,camid,embedding_path\n"
            + "\n".join(
                f"{r['sample_id']},{r['pid']},{r['camid']},{r['embedding_path']}"
                for r in rows
            )
            + "\n"
        )
    return tmp_path, data, qids, gids


def cli_input(root, mode):
    if mode == "npz":
        return [
            "--query",
            str(root / "query.npz"),
            "--gallery",
            str(root / "gallery.npz"),
        ]
    if mode in {"csv", "jsonl"}:
        return [
            "--query-manifest",
            str(root / f"query.{mode}"),
            "--gallery-manifest",
            str(root / f"gallery.{mode}"),
        ]
    return [
        arg
        for name in (
            "query_features",
            "gallery_features",
            "query_ids",
            "gallery_ids",
            "query_camids",
            "gallery_camids",
        )
        for arg in ("--" + name.replace("_", "-"), str(root / f"{name}.npy"))
    ]


@pytest.mark.parametrize("mode", ["npz", "npy", "csv", "jsonl"])
def test_all_cli_inputs_match_direct_api(inputs, mode):
    root, data, qs, gs = inputs
    expected = ReIDEvaluator(distance="cosine", query_chunk_size=1).evaluate(**data)
    output = root / f"result-{mode}"
    args = [
        "evaluate",
        *cli_input(root, mode),
        "--distance",
        "cosine",
        "--query-chunk-size",
        "1",
        "--output",
        str(output),
    ]
    assert main(args) == 0
    actual = read_result(output / "metrics.json")
    assert actual["metrics"] == expected.to_dict()["metrics"]
    assert actual["counts"] == expected.to_dict()["counts"]
    assert {p.name for p in output.iterdir()} == {
        "metrics.json",
        "config.json",
        "run_metadata.json",
        "summary.csv",
        "per_query.csv",
    }
    assert main(["inspect", str(output / "metrics.json")]) == 0


@pytest.mark.parametrize("suffix", ["json", "yaml"])
def test_config_precedence(inputs, suffix):
    root, data, qs, gs = inputs
    cfg = root / f"config.{suffix}"
    values = dict(
        protocol="custom",
        distance="euclidean",
        normalize_features=True,
        max_rank=7,
        query_chunk_size=1,
        top_k_diagnostics=1,
    )
    if suffix == "json":
        cfg.write_text(json.dumps(dict(schema_version=1, evaluation=values)))
    else:
        cfg.write_text(
            "schema_version: 1\nevaluation:\n"
            + "\n".join(
                f"  {k}: {str(v).lower() if isinstance(v, bool) else v}"
                for k, v in values.items()
            )
            + "\n"
        )
    assert load_config(cfg)["evaluation"]["distance"] == "euclidean"
    out = root / f"config-result-{suffix}"
    assert (
        main(
            [
                "evaluate",
                *cli_input(root, "npz"),
                "--config",
                str(cfg),
                "--distance",
                "cosine",
                "--no-normalize",
                "--output",
                str(out),
            ]
        )
        == 0
    )
    report = read_result(out / "metrics.json")
    assert report["evaluation"]["max_rank"] == 7
    assert report["evaluation"]["distance"] == "cosine"
    assert report["evaluation"]["normalize_features"] is False
    assert len(report["per_query"][0]["top_matches"]) == 1


def test_compatible_and_incompatible_comparison_exit_codes(inputs, capsys):
    root, data, qs, gs = inputs
    a = ReIDEvaluator(distance="cosine").evaluate(
        **data, query_sample_ids=qs, gallery_sample_ids=gs
    )
    altered = dict(data, query_features=data["query_features"] + 0.2)
    b = ReIDEvaluator(
        distance="cosine", query_chunk_size=1, top_k_diagnostics=0
    ).evaluate(**altered, query_sample_ids=qs, gallery_sample_ids=gs)
    write_result(a, root / "a")
    write_result(b, root / "b")
    assert (
        main(["compare", str(root / "a/metrics.json"), str(root / "b/metrics.json")])
        == 0
    )
    assert "unavailable" in capsys.readouterr().out
    payload = b.to_dict()
    payload["metadata"]["gallery_fingerprint"] = "changed"
    (root / "bad.json").write_text(json.dumps(payload))
    assert main(["compare", str(root / "a/metrics.json"), str(root / "bad.json")]) == 2
    assert "gallery metadata" in capsys.readouterr().err
    assert main(["compare", str(root / "a/metrics.json")]) == 2


def test_missing_population_or_protocol_evidence_is_incompatible(inputs):
    root, data, qs, gs = inputs
    a = (
        ReIDEvaluator()
        .evaluate(**data, query_sample_ids=qs, gallery_sample_ids=gs)
        .to_dict()
    )
    for key in (
        "gallery_fingerprint",
        "valid_query_fingerprint",
        "protocol_version",
        "protocol_rules",
        "metric_definitions",
    ):
        b = copy.deepcopy(a)
        b["metadata"].pop(key)
        assert compatibility_reasons(a, b)
    no_ids = ReIDEvaluator().evaluate(**data).to_dict()
    assert any("incomplete" in x for x in compatibility_reasons(no_ids, no_ids))


def test_fingerprint_deterministic_in_new_process_and_input_hashes(inputs):
    root, data, qs, gs = inputs
    code = "import numpy as np; from reid_eval_kit.hashing import array_fingerprint; print(array_fingerprint(np.arange(2000,dtype=np.int64)))"
    env = dict(os.environ)
    runs = [
        subprocess.check_output([sys.executable, "-c", code], env=env, text=True)
        for _ in range(2)
    ]
    assert runs[0] == runs[1]
    a = np.arange(4, dtype=np.int64)
    assert array_fingerprint(a) == array_fingerprint(a.astype(">i8"))
    with pytest.raises(ValueError):
        array_fingerprint(np.array([object()], dtype=object))
    a = ReIDEvaluator().evaluate(**data, query_sample_ids=qs, gallery_sample_ids=gs)
    b = ReIDEvaluator().evaluate(
        **dict(data, query_features=data["query_features"] + 0.1),
        query_sample_ids=qs,
        gallery_sample_ids=gs,
    )
    assert a.metadata["query_fingerprint"] == b.metadata["query_fingerprint"]
    assert (
        a.metadata["query_feature_fingerprint"]
        != b.metadata["query_feature_fingerprint"]
    )


def test_repeated_evaluation_and_subprocess_cli(inputs):
    root, data, qs, gs = inputs
    argv = [sys.executable, "-m", "reid_eval_kit", "evaluate", *cli_input(root, "npz")]
    metadata = []
    metrics = []
    for i in range(2):
        subprocess.run(
            [*argv, "--output", str(root / f"repeat-{i}")],
            check=True,
            capture_output=True,
            text=True,
        )
        report = read_result(root / f"repeat-{i}/metrics.json")
        metadata.append(report["metadata"])
        metrics.append(report["metrics"])
    assert metrics[0] == metrics[1]
    assert metadata[0]["query_fingerprint"] == metadata[1]["query_fingerprint"]
    assert metadata[0]["gallery_fingerprint"] == metadata[1]["gallery_fingerprint"]


@pytest.mark.parametrize(
    "text",
    [
        "[]",
        '{"evaluation":{"max_rnak":5}}',
        '{"schema_version":7}',
        '{"evaluation":{"normalize_features":"yes"}}',
    ],
)
def test_bad_json_config(tmp_path, text):
    cfg = tmp_path / "bad.json"
    cfg.write_text(text)
    with pytest.raises(InputValidationError):
        load_config(cfg)


@pytest.mark.parametrize(
    "line",
    [
        "[]",
        '{"sample_id":"a"}',
        '{"sample_id":"q","pid":1.5,"camid":1,"embedding_path":"q.npy"}',
        '{"sample_id":"q","pid":1,"camid":1,"embedding_path":"../q.npy"}',
        '{"sample_id":"q","pid":1,"camid":1,"embedding_path":"https://example.com/e.npy"}',
    ],
)
def test_malformed_manifest(tmp_path, line):
    np.save(tmp_path / "q.npy", np.ones(2))
    p = tmp_path / "bad.jsonl"
    p.write_text(line + "\n")
    with pytest.raises(InputValidationError):
        load_manifest(p)


def test_manifest_dimension_mixed_types_and_symlink_escape(tmp_path):
    np.save(tmp_path / "a.npy", np.ones(2))
    np.save(tmp_path / "b.npy", np.ones(3))
    path = tmp_path / "a.csv"
    path.write_text("sample_id,pid,camid,embedding_path\na,1,0,a.npy\nb,2,0,b.npy\n")
    with pytest.raises(InputValidationError, match="row 2.*dimension"):
        load_manifest(path)
    path.write_text(
        "sample_id,pid,camid,embedding_path\na,1,0,a.npy\nb,alice,0,a.npy\n"
    )
    with pytest.raises(InputValidationError, match="row 2.*mixed"):
        load_manifest(path)
    (tmp_path / "escape.npy").symlink_to("/tmp/outside.npy")
    path.write_text("sample_id,pid,camid,embedding_path\na,1,0,escape.npy\n")
    with pytest.raises(InputValidationError, match="escapes"):
        load_manifest(path)


def test_no_per_query_reports_and_overwrite_guard(inputs):
    root, data, qs, gs = inputs
    r = ReIDEvaluator(include_per_query=False).evaluate(**data)
    write_result(r, root / "nodetail")
    assert not (root / "nodetail/per_query.csv").exists()
    with pytest.raises(FileExistsError):
        write_result(r, root / "nodetail")
    with pytest.raises(ValueError, match="without valid"):
        bootstrap(r)


def test_invalid_files_cli_and_nonstandard_json(tmp_path):
    p = tmp_path / "metrics.json"
    p.write_text('{"schema_version":1,"metrics":{"mAP":NaN}}')
    with pytest.raises(InputValidationError):
        read_result(p)
    assert main(["evaluate", "--query", "missing.npz", "--gallery", "missing.npz"]) == 2
    assert main(["inspect", str(p)]) == 2


def test_report_refuses_symlink_overwrite(inputs):
    root, data, qs, gs = inputs
    out = root / "unsafe"
    out.mkdir()
    target = root / "original"
    target.write_text("keep")
    (out / "metrics.json").symlink_to(target)
    with pytest.raises(FileExistsError):
        write_result(ReIDEvaluator().evaluate(**data), out, overwrite=True)
    assert target.read_text() == "keep"


def test_default_index_fingerprints_are_stable_across_processes():
    code = """import numpy as np
from reid_eval_kit import ReIDEvaluator
r=ReIDEvaluator(distance="euclidean").evaluate(np.ones((1,1)),np.ones((1,1)),np.array([1]),np.array([1]),np.array([0]),np.array([1]))
print(r.metadata["query_fingerprint"], r.metadata["gallery_fingerprint"])
"""
    a = subprocess.check_output([sys.executable, "-c", code], text=True)
    b = subprocess.check_output([sys.executable, "-c", code], text=True)
    assert a == b


def test_malformed_yaml_and_config_input_overwrite_guard(tmp_path, inputs):
    root, data, qs, gs = inputs
    bad = tmp_path / "broken.yaml"
    bad.write_text("evaluation: [")
    with pytest.raises(InputValidationError, match="YAML"):
        load_config(bad)
    (root / "config.json").write_text(
        json.dumps({"evaluation": {"distance": "cosine"}})
    )
    original = (root / "config.json").read_text()
    assert (
        main(
            [
                "evaluate",
                *cli_input(root, "npz"),
                "--config",
                str(root / "config.json"),
                "--output",
                str(root),
                "--overwrite",
            ]
        )
        == 2
    )
    assert (root / "config.json").read_text() == original


def test_invalid_query_fields_and_top_k_never_changes_metric_population(inputs):
    root, data, qs, gs = inputs
    data = dict(data, query_ids=np.array([1, 999]))
    a = ReIDEvaluator(top_k_diagnostics=0).evaluate(**data)
    b = ReIDEvaluator(top_k_diagnostics=4).evaluate(**data)
    assert a.to_dict()["metrics"] == b.to_dict()["metrics"]
    assert a.per_query[1].rank10_success is None
    assert a.per_query[1].top_matches == []
    assert len(b.per_query[1].top_matches) == 4


def test_npz_npy_content_mismatch_and_pickle_rejected(tmp_path):
    from reid_eval_kit.io import load_archive
    from reid_eval_kit.cli import _arr
    from reid_eval_kit.exceptions import EvaluationError

    path = tmp_path / "pretend.npz"
    with path.open("wb") as f:
        np.save(f, np.ones((1, 1)))
    with pytest.raises(InputValidationError, match="expected an NPZ"):
        load_archive(path)
    path = tmp_path / "pretend.npy"
    with path.open("wb") as f:
        np.savez(f, features=np.ones((1, 1)))
    with pytest.raises(EvaluationError, match="single NPY"):
        _arr(path)
    manifest = tmp_path / "input.csv"
    manifest.write_text("sample_id,pid,camid,embedding_path\nq,1,0,pretend.npy\n")
    with pytest.raises(InputValidationError, match="row 1.*single NPY"):
        load_manifest(manifest)
    obj = tmp_path / "object.npz"
    np.savez(
        obj,
        features=np.ones((1, 1)),
        pids=np.array(["a"], dtype=object),
        camids=np.array([0]),
    )
    with pytest.raises(InputValidationError, match="allow_pickle=False"):
        load_archive(obj)


def test_report_rejects_corrupt_cmc_and_counts(inputs):
    root, data, qs, gs = inputs
    original = ReIDEvaluator().evaluate(**data).to_dict()
    for change in ("nonfinite", "decreasing", "unavailable", "counts"):
        payload = copy.deepcopy(original)
        if change == "nonfinite":
            payload["metrics"]["cmc"][0] = float("inf")
        elif change == "decreasing":
            payload["metrics"]["cmc"][:2] = [1.0, 0.0]
        elif change == "unavailable":
            payload["metrics"]["cmc"][10] = 1.0
        else:
            payload["counts"]["num_valid_queries"] = 0
        path = root / f"{change}.json"
        path.write_text(json.dumps(payload))
        with pytest.raises(InputValidationError):
            read_result(path)

# reid-eval-kit

A NumPy-based CPU toolkit for evaluating precomputed person re-identification
embeddings. It shares one engine between the Python API and `reideval` CLI,
with multishot CMC, mAP, mINP, stable ranking, query diagnostics, and reproducible
JSON/CSV reports. Evaluate existing embeddings, inspect retrieval failures,
and compare experiments on the same query/gallery population.

## Installation

Python 3.11 or newer is required. From the root of this source checkout,
create a virtual environment using Python's built-in `venv` module.

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Then install with `pip` in the activated environment:

```bash
python -m pip install --upgrade pip
python -m pip install '.[yaml]'
python -m reid_eval_kit --help
```

NumPy is the only mandatory runtime dependency. The optional `yaml` extra adds
PyYAML; install with `python -m pip install .` if JSON configuration is enough.
The package has not been published to PyPI. See [Getting started](docs/getting-started.md)
for wheel installation, offline installation, and troubleshooting.

## Python API

This complete example uses synthetic vectors:

```python
import numpy as np
from reid_eval_kit import ReIDEvaluator

result = ReIDEvaluator(protocol="custom", distance="cosine").evaluate(
    query_features=np.array([[1.0, 0.0]]),
    gallery_features=np.array([[1.0, 0.0], [0.0, 1.0]]),
    query_ids=np.array([1]),
    gallery_ids=np.array([1, 2]),
    query_camids=np.array([0]),
    gallery_camids=np.array([1, 1]),
    query_sample_ids=np.array(["q0"]),
    gallery_sample_ids=np.array(["g0", "g1"]),
)
print(result.rank1, result.mAP, result.mINP)
assert result.rank5 is None  # only two gallery samples
```

Advanced callers can pass `EvaluationInput` via `evaluate(inputs=data)` and
construct `ReIDEvaluator(config=EvaluationConfig(...))`. All scores are fractions
in [0,1]. Rank positions in diagnostics are one-based; gallery indices are
zero-based. No-positive queries are diagnosed and excluded from aggregate
means; an all-invalid evaluation raises an exception.

## CLI quickstart

Run from the source checkout after installation. All inputs are generated
locally; no research dataset or model is downloaded.

```bash
python examples/generate_synthetic_data.py
```

Evaluate the synthetic embeddings and inspect the saved report. The following
multiline commands use Bash syntax; in PowerShell, enter each command on one line.

```bash
reideval evaluate \
  --query synthetic/query.npz \
  --gallery synthetic/gallery.npz \
  --protocol custom \
  --distance cosine \
  --normalize \
  --output results/example
reideval inspect results/example/metrics.json
```

Use fresh output directories, or `--overwrite` explicitly. Reports contain
`metrics.json`, `summary.csv`, `config.json`, and `run_metadata.json`; per-query
retention also writes `per_query.csv`. Saved top-k diagnostics never truncate
the full ranking used for AP/INP. Stable sample IDs are required for comparison;
changed embeddings are allowed, changed populations/protocols are rejected.

NPZ archives require `features`, `pids`, and `camids`; optional `sample_ids`
improve diagnostics and establish sample correspondence. Separate NPY and
CSV/JSONL manifests are also supported. See [input contracts](docs/input-formats.md).

## Evaluation protocols

| Protocol | Main rules | Verification scope |
|---|---|---|
| `market1501` | Exclude gallery PID `-1`; retain background PID `0`; exclude same-person/same-camera entries; cameras `0..5`. | Compatibility with the inspected Torchreid loader-plus-ranking pipeline. |
| `msmt17` | Nonnegative integer IDs; cameras `0..14`; same-person/same-camera exclusion; caller-supplied query/gallery splits. | **Experimental.** Source-verified FastReID metadata and ranking behavior; official split files were not checked. |
| `custom` | Integer or string identifiers with configurable junk IDs and same-person/same-camera exclusion. | Explicit caller-defined rules. |

[Protocol evidence](docs/protocols.md) records examined sources and unresolved
claims. Named protocol rules cannot be overridden. Use the effective protocol
version and rules in scientific reports; canonical MSMT17 end-to-end equivalence
has not been validated.

## Metrics

CMC rank-k is the fraction of valid queries with a positive in the first k
filtered candidates. AP averages precision at every positive rank; mAP averages
AP across valid queries. INP is positives divided by last-positive rank;
mINP averages INP. Rank-k emphasizes the first positive while mAP and mINP
also reflect later positive retrieval. See [validated equations](docs/metrics.md).
Different exclusions or query populations make paper results noncomparable.

## Reproducibility and limitations

Exact ties retain original gallery order. Reports record configuration,
versions, protocol rules, feature and metadata hashes, and invalid-query counts.
Query chunking bounds distance-matrix storage but still sorts full rankings.
No cross-platform bitwise guarantee is made near numerical ties.
[Reproducibility](docs/reproducibility.md), [comparison](docs/comparison.md), and
[limitations](docs/limitations.md) explain precision and statistical limits.
The optional bootstrap utility provides seeded marginal query-level percentile
intervals, not paired significance tests or identity-cluster inference.

## Comparing runs

To check that query chunking preserves results, evaluate the same synthetic
inputs in a fresh output directory:

```bash
reideval evaluate \
  --query synthetic/query.npz \
  --gallery synthetic/gallery.npz \
  --protocol custom \
  --distance cosine \
  --normalize \
  --query-chunk-size 1 \
  --output results/repeat
reideval compare results/example/metrics.json results/repeat/metrics.json
```

Model comparison uses the same command with reports from different embeddings.
Stable sample IDs, population metadata, and protocol rules must agree. See
[comparison requirements](docs/comparison.md) for compatibility checks.

## Development

In the virtual environment created above, install the development extra:

```bash
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
mypy src/
python -m pytest -ra
python -m build
```

See [CONTRIBUTING](CONTRIBUTING.md) for the full development and packaging
workflow, and the [release audit](docs/release-audit.md) for executed checks.

## Citation

When citing results, cite the dataset publications and mINP reference linked in
[protocol provenance](docs/protocols.md), this toolkit's version, effective
protocol identifier, preprocessing, ranking policy, and query validity counts.
No benchmark accuracies or dataset results are bundled.

## License

Original code is under the [MIT license](LICENSE).
[Dependency license notes](docs/licenses.md) describe reviewed upstream and
runtime dependencies.

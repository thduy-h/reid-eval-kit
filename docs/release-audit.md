# v0.1.0 correctness and release audit

Audit date: 2026-10-27. **CONDITIONAL GO for the documented compatibility scope;
public release remains unapproved.** Metric correctness and local package use
are verified below. No canonical MSMT17 split-validation claim is authorized.
No tag, push, publication, or commit occurred.

## Scientific defects found and corrected

1. Default sample indices used object arrays, whose bytes contain process-local
   addresses. Version-2 fingerprints use non-object values, explicit framing,
   and little-endian representation. Separate feature hashes now accompany
   population and valid-query fingerprints.
2. Invalid-query positional construction assigned `top_matches` to
   `rank10_success`. All result fields now align; invalid queries retain
   detailed reasons and diagnostics without participating in aggregate means.
3. CMC beyond the gallery domain incorrectly remained numeric. It is now null,
   as are unavailable per-query named ranks. Named rank-5/10 no longer depend
   on whether the saved CMC maximum is smaller than those ranks.
4. Euclidean cancellation could erase a small distance between huge vectors;
   naive norms also underflowed/overflowed. Direct stable fallback and scaled
   normalization handle these cases, rejecting unrepresentable distances.
   Cosine normalization occurs once. Fixed contractions preserve chunk
   reductions and stable exact ties on the tested platform.
5. MSMT17 inherited an undocumented -1 junk sentinel. Inspected loaders and
   evaluator define none; the experimental adapter now has no global junk and
   rejects negative IDs. Named versions changed when semantics changed.
6. Reports recorded configured rules even if a named protocol ignored them.
   Named overrides now fail, and reports record actual effective rules. Strict
   identifier/camera validation prevents accidental label coercion.
7. Comparison checked mostly counts/configuration and could crash on null
   rank deltas or return success for incompatibility. It now requires stable
   population IDs/fingerprints, valid population, schema, protocol, rules,
   metric/tie definitions, and emits no deltas for incompatible runs (exit 2).
   Feature hashes may differ; execution/diagnostic controls may differ.
8. CLI option detection generated names prefixed with underscores, allowing
   explicit flags to be overwritten by config. Correct precedence is tested
   for JSON/YAML and explicit normalization disable. Unsafe permissive YAML
   parsing was replaced with optional safe PyYAML and strict schema/key checks.
9. Manifest parsing could coerce mixed identifiers and allow unrestricted
   paths; file content/extension mismatches could leak raw attribute errors.
   Inputs now reject these cases, pickle, missing data, duplicate samples,
   inconsistent dimensions, remote paths, and path/symlink escapes.
10. Reports lacked atomic writes and artifact validation; stale/symlink outputs
    could be overwritten. Writes are per-file atomic, explicit overwrite is
    guarded, and report readers reject malformed counts, metric domains, CMC,
    schema, and nonstandard NaN/Infinity. No-detail runs omit per-query CSV.
11. The source archive omitted oracle/documentation assets, build requirements
    were too old for SPDX metadata, and CI did not install its quality tools.
    Packaging/manifest/CI were corrected and independently installed.

AP uses the count of surviving positives as denominator; INP uses the
one-based last-positive rank. Those original formulas were correct and were
retained, then verified independently. CMC means use valid queries only and
extend the step function across shorter candidate lists.

## Protocol evidence

[protocols.md](protocols.md) records the inspected Torchreid Market/MSMT loaders,
Torchreid rank kernel, FastReID MSMT loader and evaluation call path, FastReID
INP kernel, and Open-ReID discussion. Market -1 loader removal, background 0
retention, and per-query camera exclusion were directly read. MSMT split names,
15-camera zero-based conversion, and the consistent FastReID PID offset were
read independently. These were static source reviews, not executed upstream
framework cross-checks; no PyTorch framework or restricted data was installed.
`master` was retrievable via the web reader but immutable commit IDs were not.
Official dataset pages/paper pages and the DOI full text were blocked; arXiv
bibliography was accessible. These limits remain explicit, not marked verified.

## Executed quality and correctness checks

Local tools: Python 3.11.16, NumPy 2.4.4 (Nix), pytest 9.0.3,
Ruff 0.15.14, mypy 1.20.1. Tools and type stubs were made available through the
local Nix store/builds. PYTHONPATH exposed this checkout and those development
dependencies for source tests; unrelated pytest plugin autoload was disabled.

| Command | Observed result |
|---|---|
| `ruff check .` | `All checks passed!` |
| `ruff format --check .` | `21 files already formatted` |
| `mypy src/` | `Success: no issues found in 12 source files` |
| `pytest -ra` | **142 passed, 0 failed, 0 skipped** |
| `python -m build --wheel --sdist --no-isolation` | wheel and source archive built successfully using locally supplied build dependencies |

28 tests are under `tests/integration/`; 114 are other correctness/unit checks.
The independent oracle lives in `tests/reference/reference_eval.py` and uses
scalar Python/math distance equations, explicit filtering, and loop formulas,
without production helpers. Hand fractions, randomized oracle comparisons,
multiple/single positives, invalid populations, both dataset variants/custom
rules, exact ties, varying candidate counts, float dtypes, numerical extremes,
chunking, every loader, malformed input, API/CLI parity, reports, comparison,
config precedence, and nondegenerate seeded bootstrap are exercised. Coverage
was not measured. Python 3.12/3.13 CI is configured but not locally executed;
the workflow's YAML structure and least-privilege configuration were checked.

## Numerical measurements

A separate executed scalar-oracle audit used 8 seeds × 2 metrics × 2 dtypes,
with 7 queries, 21 gallery vectors, dimension 5. These are validation inputs,
not real-dataset model benchmark scores.

| dtype / metric | Maximum absolute distance discrepancy | Maximum relative distance discrepancy |
|---|---:|---:|
| float32 / Euclidean | 8.881784197001252e-16 | 4.2841925329097713e-16 |
| float32 / cosine | 4.440892098500626e-16 | 3.2226438203398937e-15 |
| float64 / Euclidean | 1.3322676295501878e-15 | 1.1873162171031746e-15 |
| float64 / cosine | 4.440892098500626e-16 | 5.88628904984754e-15 |

Maximum observed aggregate AP/INP discrepancy was **0.0** in those 32 cases.
Query chunks 1, 2, 7, and 40 produced exactly equal distance arrays in that
measurement. Separate tests cover exact/near ties, zero norms, scaled vectors
from 1e-300 to 1e300, and large cancellation. These observations do not prove
bitwise equivalence across hardware or NumPy versions. No performance benchmark
or peak-memory measurement was executed.

## Packaging and isolated runtime

Both artifacts are in `dist/`. Wheel metadata was inspected for package/import
name, version 0.1.0, Python >=3.11, MIT license, NumPy-only mandatory runtime,
optional YAML extra, console entry point, and included license. Every wheel
Python source matched the checkout. The source archive includes the tests,
oracle, docs, configs, examples, and workflow.

A fresh `/tmp/reid-wheel-venv` installed the local wheel with locally supplied
NumPy 2.4.6 and optional PyYAML 6.0.3 wheels. Dependency wheels were repacked
from actual cached/Nix distributions, not synthesized implementations. Import
and CLI smoke checks ran from `/tmp`, with PYTHONPATH cleared and Python's
isolated flag. Import resolved to the venv's site-packages, not `src/`.
`pip check` reported `No broken requirements found.` On NixOS the NumPy binary
required system `libstdc++.so.6` and `libz.so.1`; the existing GCC/zlib library
directories were exposed via LD_LIBRARY_PATH. This runtime requirement is
reported rather than hidden as a development import.

Installed-wheel tests performed direct arrays, NPZ, separate NPY, CSV, JSONL,
YAML config, JSON/CSV outputs, inspect, compatible comparison, incompatible
comparison (exit 2), overwrite rejection, repeated evaluation and chunk-size
parity. A second isolated environment installed from the source archive with
isolated build dependency provisioning from local wheels; source installation
with the optional YAML extra also succeeded. README Python and synthetic CLI
examples were executed. Source tests validate malformed inputs without requiring
any restricted data. Fresh output directories were used.

The executed default_rng(42) synthetic example used 4 queries, 12 gallery
vectors and dimension 8: rank-1 0.0, rank-5 0.5, rank-10 1.0, mAP/mINP
0.23849206349206348. Repeating with one-query chunks reproduced these metrics.
These are synthetic retrieval outcomes only. Development extras installation
and `uv sync` could not be completely reproduced offline (Twine metadata was
missing); actual local quality-tool execution above succeeded via Nix. Twine
upload/check was not executed; metadata/archive checks were executed directly.

## Release conditions and Git

Core correctness, API/CLI parity, quality tools, distributions, isolated runtime,
and documented compatibility behavior have passing local evidence. Remaining
conditions: no universal `uv.lock` (offline resolution lacks Twine), no pinned
upstream commit, no official MSMT17 split-file validation, and no Git commit.
No canonical MSMT17 claim is allowed; experimental status is already recorded.
These limits must remain disclosed. Public release still requires the owner's
authorization; this audit does not publish anything.

`git rev-parse --show-toplevel`, `git status`, and `git worktree list` all report
that the directory is not a Git repository. `.git` is an **empty tmpfs mounted
read-only**, owned by UID 1000/group users, mode 0555; it has no HEAD, config,
objects, or worktree metadata. No repository metadata was deleted, replaced,
or chmodded. Initialization and an initial commit are impossible in this mount.
There is no current branch/commit/tag to report. Files below exist as workspace
changes; Git cannot classify them until the mount is writable or the source is
copied into a valid repository. No push/tag/publication occurred.

## Exact modified or added source files

Formatting-only changes are included because no valid Git baseline is present.
Generated caches, egg-info, and ignored build directories are excluded.

```text
.github/workflows/ci.yml
.gitignore
CHANGELOG.md
CONTRIBUTING.md
MANIFEST.in
README.md
SECURITY.md
pyproject.toml
examples/generate_synthetic_data.py
docs/comparison.md
docs/input-formats.md
docs/licenses.md
docs/limitations.md
docs/metrics.md
docs/protocols.md
docs/release-audit.md
docs/reproducibility.md
src/reid_eval_kit/__init__.py
src/reid_eval_kit/__main__.py
src/reid_eval_kit/analysis.py
src/reid_eval_kit/api.py
src/reid_eval_kit/cli.py
src/reid_eval_kit/config.py
src/reid_eval_kit/exceptions.py
src/reid_eval_kit/hashing.py
src/reid_eval_kit/io.py
src/reid_eval_kit/metrics.py
src/reid_eval_kit/protocols.py
src/reid_eval_kit/types.py
tests/__init__.py
tests/integration/test_workflows.py
tests/reference/reference_eval.py
tests/test_hardening.py
tests/test_metrics.py
tests/unit/test_numerical_validation.py
tests/unit/test_protocols.py
tests/unit/test_reference.py
```

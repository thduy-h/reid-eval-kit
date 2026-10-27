# Reproducibility

Keep query/gallery ordering and stable sample IDs fixed. Exact equal computed
distances retain original gallery index order; nearly equal distances are
ordered numerically. Original indices are zero-based; external rank positions
are one-based. Input sample ordering is never randomized.

Reports include effective configuration, actual protocol rules and version,
metric/ranking definitions, valid-query population, feature shapes and original
dtypes, Python/NumPy/platform metadata, timestamp, and separate input-feature
and population fingerprints. No personal paths, usernames, or embedding values
are logged. Timestamps are intentionally different between separate runs.

Fingerprint version 2 uses SHA-256 with a domain prefix and per-array length
frames, a JSON dtype/shape header, and C-contiguous little-endian values. Unicode
arrays use NumPy's fixed-width Unicode representation; object arrays are
rejected. This removes the former nondeterminism from Python object-pointer
bytes. Metadata fingerprints include ordered PIDs, camera IDs, and sample IDs;
feature fingerprints describe the float64 arrays actually evaluated. Absent
sample IDs use ordinal indices for diagnostics but are explicitly flagged;
comparison requires supplied stable IDs to establish population equivalence.
Feature hashes are not required to match between models.

NumPy contractions have a fixed reduction order across query chunk sizes in
this implementation. Hardware and numerical-library differences can affect
near ties; exact bitwise equality across all platforms is not guaranteed.
See [metrics](metrics.md) for precision and rank availability.

The synthetic example uses `default_rng(42)` and no external dataset. Bootstrap
uses its own explicit seed. Ordinary query bootstrap can underestimate
uncertainty when identities contribute multiple dependent queries. It is a
percentile CI for AP/INP means, not a significance test; a one-query population
produces a degenerate interval. Paired differences and identity-cluster
bootstrap are not provided. Do not substitute independent marginal intervals
for a paired comparison.

Use the standard `venv` and `pip` workflow in
[Getting started](getting-started.md). Save `python --version` and
`python -m pip freeze` alongside the report to record the installed environment.
Dependencies currently have lower bounds rather than a complete committed
lockfile, so fresh installations may resolve different versions. An installed
environment snapshot is not a cross-platform lock. Development tools and
runtime versions actually used in the audit are recorded in
[release-audit](release-audit.md).

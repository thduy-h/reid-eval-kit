# Local input contracts

NPZ query and gallery archives each contain `features` (N,D), `pids` (N,),
`camids` (N,), and optional unique `sample_ids` (N,). Features must be nonempty,
finite real arrays with matching D. float32 and float64 are supported; integers
are converted to float64. Metadata must be consistently typed integers or
Unicode strings (custom protocol); floats, booleans, object/complex arrays,
empty identifiers, and mixed-type sequences are rejected. Named dataset
protocols require integer IDs and zero-based cameras as documented in
[protocols](protocols.md). Sample IDs are unique per split; the same name may
occur in different splits, since the toolkit does not infer self-exclusion.

All NumPy loading uses `allow_pickle=False`. There is no pickle fallback.
Separate NPY files require all six feature/PID/camera paths, for example:

```bash
reideval evaluate --query-features query_features.npy --gallery-features gallery_features.npy --query-ids query_ids.npy --gallery-ids gallery_ids.npy --query-camids query_camids.npy --gallery-camids gallery_camids.npy --protocol custom --output results-npy
```

CSV or JSONL manifests are separate for query and gallery. Required fields are
`sample_id`, `pid`, `camid`, `embedding_path`; optional metadata fields are
ignored by metric computation, including `image_path`. Image contents are
never loaded. Each embedding path references one one-dimensional NPY vector.
Row order is ranking input order; duplicate sample IDs and inconsistent
embedding dimensions fail with row context.

```csv
sample_id,pid,camid,embedding_path
q0,12,0,embeddings/q0.npy
```

CSV canonical decimal PID/camera tokens are converted to integers; other tokens
stay exact strings. Leading-zero tokens such as `001` stay strings. Mixing
converted numeric and textual identifiers within one field is rejected. JSONL
requires JSON integers or strings and preserves their type. Embeddings are not
stored inside JSON.

Paths must be local, relative, and remain inside the manifest directory after
resolving symlinks. Absolute paths, URLs, and parent traversal out of the root
are rejected. Place manifests at the intended dataset root. Loading arrays and
sorting exact full rankings require memory proportional to the input arrays
and one query chunk times gallery size; there is no hostile-file allocation
quota. Process untrusted very large archives with OS-level resource limits.

Configuration accepts JSON, or YAML through `pip install '.[yaml]'`.
`schema_version: 1` and an `evaluation` object are supported; its keys match
`EvaluationConfig`. Unknown sections/keys and malformed types are errors.
CLI explicitly supplied flags override config; config overrides library/CLI
defaults. `--no-normalize` explicitly disables config normalization. Advanced
experiment/output/bootstrap configuration sections are not implemented.

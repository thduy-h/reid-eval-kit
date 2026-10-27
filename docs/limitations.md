# Limits of v0.1.0

This toolkit evaluates precomputed embeddings on CPU. No training, extraction,
GPU execution, reranking, approximate search, dataset download, or dataset
redistribution is included. HTML reports are not implemented.

Market-1501 is a named Torchreid loader-plus-rank compatibility variant;
MSMT17 is an experimental metadata compatibility variant with a source-verified
ranking kernel. Official restricted split files were not checked. Immutable
upstream source commit IDs were not resolved. Do not claim canonical dataset
benchmark validation or equivalence to all published evaluation variants.

Exact AP and INP require the full filtered ranking, even when diagnostics save
only top-k. Query chunking reduces distance memory, not full-gallery sorting
complexity. Inputs are held in memory and promoted to float64; each query chunk
uses O(chunk_size * gallery_size) distance storage and each sort O(Ng log Ng)
time. The default automatic chunk size is at most 128, while an explicit size
can request a larger block. Retained diagnostics grow with query count and
top-k; no constant-memory or measured performance claim is made.

Per-file report writes are atomic, but the directory is not a multi-file
transaction against process termination or concurrent writers. Use distinct
output directories and avoid concurrent overwrite. No allocation quota is
imposed on malformed gigantic files. Strict paths reject symlink escapes.

Comparison requires stable IDs and exact ordering; it does not realign runs.
Bootstrap is marginal query-level percentile uncertainty only; dependent
identities, paired model-delta inference, and significance tests are not
implemented. Small populations can yield uninformative intervals.

No cross-platform bitwise promise is made for floating-point near ties.
Dependencies are not pinned by a complete committed lockfile. See
[release-audit](release-audit.md) for executed checks and remaining limitations.

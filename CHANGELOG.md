# Changelog

## 0.1.0 — unreleased

- NumPy CPU evaluation with multishot CMC, positive-rank AP/mAP, last-positive
  INP/mINP, exact stable ranking, and row chunking.
- Explicit Market-1501 loader/rank compatibility and experimental MSMT17
  metadata compatibility; custom junk and camera exclusions.
- NPZ, separate NPY, CSV/JSONL input; safe JSON/YAML configuration; API/CLI
  parity; diagnostics; JSON/CSV reports; guarded experiment comparison.
- Seeded marginal query bootstrap with documented statistical limitations.
- Hardening fixes: deterministic pointer-free fingerprints, population and
  protocol comparison checks, nonzero incompatible exit status, null unavailable
  ranks, invalid-query field alignment, large/small-norm numerical stability,
  explicit CLI/config precedence, strict metadata types, safe manifest paths,
  atomic per-file reports and overwrite guards, and source-archive test assets.
- Independent scalar correctness tests, numerical and integration regressions,
  local Ruff/mypy checks, and installed-distribution audit.

No release/tag or publication has been performed. Protocol IDs changed to v2
where sentinel or validation semantics changed. See the audit for evidence and
remaining release conditions.

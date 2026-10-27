# Comparing evaluations

`reideval compare A/metrics.json B/metrics.json` checks schema/tool versions,
protocol version and actual rules, metric and tie definitions, scientific
configuration, query/gallery metadata fingerprints, and the valid-query
population. Explicit stable query and gallery sample IDs are required;
ordinal indices alone provide incomplete evidence of sample composition.
Absent or mismatched evidence causes a precise error and nonzero exit status,
with no improvement claim. Embedding hashes may differ: that is the intended
comparison between models. Chunk size and diagnostic retention do not affect
compatibility. No sorting or identity-level realignment is inferred.

For compatible runs, every delta is `candidate - baseline`, in fractional
units; multiply by 100 for percentage points. Unavailable ranks stay
unavailable, including their deltas. A positive delta alone does not establish
statistical significance. The CLI supports a baseline and several candidates,
checks them all before reporting deltas, and does not recompute metrics.

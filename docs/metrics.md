# Validated metric definitions

Let `r(k)` be binary relevance after all protocol exclusions, with rank `k`
starting at one. Let `M = sum(r(k))`. A query is valid only when `M > 0`.

- `P(k) = sum(r(j), j=1..k) / k`.
- `AP = sum(P(k) * r(k)) / M`. No trapezoidal interpolation is used.
- `CMC(k) = 1` if the first eligible positive occurs at or before `k`, else 0.
- `INP = M / rank(last eligible positive)`.
- `mAP`, `mINP`, and CMC average over valid queries, with equal query weight.

For `[1,0,1,0]`: AP is `5/6`, INP `2/3`, first positive rank 1, last rank 3.
For `[0,1,0,1]`: AP and INP are `1/2`, first rank 2, last rank 4. With one
positive, AP and INP both equal its reciprocal rank. Multiple positive matches
count individually; no single-gallery-shot random sampling occurs.

Queries with no eligible positive are excluded from aggregation and retain a
specific reason (`identity_absent`, `no_eligible_positive`,
`empty_filtered_gallery`, or `junk_query_identity`). This differs from a valid
query whose positive is ranked poorly. If every query is invalid, evaluation
raises `EvaluationError`; no zero-valued aggregate is invented.

Each valid query's CMC remains 1 once its positive has appeared, including after
its filtered candidate list ends. The global supported domain is ranks 1..Ng,
where Ng is the **unfiltered** gallery count. Beyond Ng, CMC entries and named
rank metrics are `None` / JSON `null`. `max_rank` limits the saved CMC vector;
named rank-1/5/10 values are computed independently within the supported domain.
Per-query named rank diagnostics use the same domain.

Tests use hand fractions, random independent scalar oracles, exact ties,
different candidate counts, invalid populations, and query chunk sizes. See
[protocols](protocols.md) for source comparisons and implementation differences.

# Distance precision

Features are validated and computed in float64 (float32 inputs are promoted,
without recovering precision already lost in storage). Original dtypes are
recorded. Cosine inherently normalizes once even when optional preprocessing
is disabled, and clips similarity to [-1,1]. Scaling by the maximum absolute
component before the L2 norm avoids overflow/underflow. Only exactly zero
vectors are rejected; no arbitrary small-norm threshold is used.

Euclidean uses the squared-norm/dot-product identity. Cancellation-prone pairs
(with squared distance at most `64 * eps * (norm_q_squared + norm_g_squared)`),
subnormal squared norms, or nonfinite intermediate values are recomputed by
direct subtraction and `hypot.reduce`, in bounded gallery blocks. Exactly zero
vectors are valid for unnormalized Euclidean. Unrepresentable distances fail
with a rescaling error. Optional normalization changes Euclidean rankings and
is always recorded.

A fixed NumPy contraction order makes changing query chunk size preserve
computed distance rows on the tested platform, including exact ties. There is
no approximate tie collapsing. Across platforms or NumPy versions, near ties
may still cross; bitwise cross-platform reproducibility is not promised.

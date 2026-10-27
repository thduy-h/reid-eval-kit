# Protocol provenance and compatibility

Reviewed on 2026-10-27. Direct shell downloads failed DNS resolution; the web
reader successfully returned the upstream files below. Branch references are
`master` as retrieved, **not pinned commits**: GitHub commit resolution was
unavailable. No upstream implementation is vendored or executed in this suite.
The tests implement a separate scalar oracle and check the equations against
these inspected source functions. A source review is distinct from evaluation
of restricted dataset files, which were not available or downloaded.

| Area / dataset | Publication and dataset documentation | Reference implementation / relevant function | Version | Evidence and verification status |
|---|---|---|---|---|
| Market-1501 | Zheng et al., *Scalable Person Re-identification: A Benchmark*, ICCV 2015; [paper](https://openaccess.thecvf.com/content_iccv_2015/html/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.html), [dataset project](http://www.liangzheng.org/Project/project_reid.html) | [Torchreid Market loader](https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/data/datasets/image/market1501.py), `Market1501.process_dir` | master, reviewed 2026-10-27 | Source inspected: removes -1, retains 0 as background; validates 0..1501 and cameras 1..6, then subtracts 1 from cameras. Query/gallery labels are not relabeled. Official publication/project pages were not independently accessible. |
| Market-1501 ranking | [Open-ReID evaluation discussion](https://cysu.github.io/open-reid/notes/evaluation_metrics.html) | [Torchreid rank.py](https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/metrics/rank.py), `eval_market1501` | master, reviewed 2026-10-27 | Source and discussion inspected: same-ID AND same-camera exclusion, skip no-positive queries, first-positive CMC, positive-rank precision mean AP. Rank function does not itself remove global junk; this relies on loader filtering. |
| MSMT17 metadata | Wei et al., *Person Transfer GAN to Bridge Domain Gap for Person Re-Identification*, CVPR 2018; [paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Wei_Person_Transfer_GAN_CVPR_2018_paper.html), [dataset project](http://www.pkuvmc.com/publications/msmt17.html) | [Torchreid MSMT17](https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/data/datasets/image/msmt17.py), `process_dir`; [FastReID MSMT17](https://github.com/JDAI-CV/fast-reid/blob/master/fastreid/data/datasets/msmt17.py), `process_dir` and `__init__` | master, reviewed 2026-10-27 | Both sources inspected: list_query.txt/list_gallery.txt define evaluation splits; V1 train/test versus V2 mask_train_v2/mask_test_v2, filename camera token minus one, 15 cameras. Torchreid preserves list PIDs; FastReID adds the same training-count offset to query/gallery PIDs. Neither defines a negative junk sentinel. Dataset files and official pages unverified. |
| MSMT17 ranking / INP | Ye et al., *Deep Learning for Person Re-Identification: A Survey and Outlook*, [DOI](https://doi.org/10.1109/TPAMI.2021.3054775), [arXiv v2](https://arxiv.org/abs/2001.04193v2) | [FastReID rank.py](https://github.com/JDAI-CV/fast-reid/blob/master/fastreid/evaluation/rank.py), `eval_market1501`; [ReidEvaluator.evaluate](https://github.com/JDAI-CV/fast-reid/blob/master/fastreid/evaluation/reid_evaluation.py) | master, reviewed 2026-10-27 | Source inspected: evaluator passes query/gallery metadata to the multishot rank kernel; INP is cumulative positives at last-positive index divided by index+1, then averaged. No global sentinel filtering there. DOI full text blocked; arXiv bibliographic record inspected. |

## Implemented rules and decisions

`market1501` identifies **`market1501-torchreid-loader-rank-v2`**, compatibility
with the inspected loader-plus-rank pipeline, not every evaluation script ever
used for Market-1501. It removes gallery PID `-1`, retains background PID `0`,
and removes samples sharing both query PID and camera. Different-person
same-camera samples remain negatives. `0` is not globally junk in this variant.
The loader's `_junk_pids = [0, -1]` declaration is not used by `process_dir`
to remove background: the actual method only skips `-1`. A raw rank function
called with unfiltered `-1` gallery entries differs from the composed pipeline.
Our masks handle gallery junk before ranking. Query `-1` is invalid; other
unsupported negative PIDs are rejected. Cameras must already be 0..5.

`msmt17` identifies **`msmt17-fastreid-metadata-v2`** and is labeled
**experimental: supplied splits, source-verified kernel** in every report.
There is no invented `-1` sentinel: negative PIDs are rejected and no global
junk is removed. PID `0` is an ordinary identity. Cameras must already be
0..14. Query/gallery must come from the supplied official split lists; the
package cannot prove their provenance from embeddings. No claim of canonical
end-to-end dataset equivalence or actual MSMT17 benchmark validation is made.
The FastReID joint test PID offset preserves equality; it is not necessary to
undo it if applied consistently. Applying different maps independently to query
and gallery corrupts evaluation.

`custom` allows integer or string IDs with explicit `junk_ids` and
`exclude_same_id_same_camera`. Use it for alternate label conventions or
protocols. Features never determine sample identity. Named protocols reject
custom rule overrides rather than silently ignoring them. Every result records
effective rules, version, validity population, and ranking policy. Protocol
versions were changed because v1 incorrectly invented MSMT17 junk semantics.

No-positive queries are omitted from aggregates and retained in diagnostics.
All-invalid evaluation raises an exception. CMC extends each valid query's
step function over the unfiltered gallery domain; ranks beyond that domain
are unavailable. Upstream NumPy sorts do not promise our exact stable tie
policy, and some upstream implementations assume equal-length CMC slices.
We explicitly use original gallery order for exact ties and support variable
candidate counts. These are documented differences, not canonical-equivalence
claims. Other-camera same-person samples remain positives.

Original IDs are required by the strict Market adapter (-1..1501); consistent
renaming is supported by custom. MSMT17 accepts nonnegative integer IDs,
including consistent FastReID offsets. The toolkit never rewrites labels or
cameras, infers splits, downloads data, or treats loader code alone as proof of
an evaluation protocol. Unresolved: immutable upstream commit IDs, official
split-file contents, canonical MATLAB junk annotations, and real-dataset
end-to-end parity. Use the named compatibility scope when citing results.

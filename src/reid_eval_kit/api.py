from __future__ import annotations
import platform
import sys
import time
import numpy as np
from .types import EvaluationConfig, EvaluationInput, EvaluationResult, QueryResult
from .metrics import validate_features, validate_vector, distances, stable_order
from .protocols import resolve_protocol
from .exceptions import EvaluationError, InputValidationError
from .hashing import array_fingerprint


class ReIDEvaluator:
    def __init__(
        self,
        protocol="custom",
        distance="cosine",
        normalize_features=False,
        max_rank=50,
        query_chunk_size=None,
        junk_ids=(-1,),
        exclude_same_id_same_camera=True,
        top_k_diagnostics=10,
        include_per_query=True,
        *,
        config: EvaluationConfig | None = None,
    ):
        if config is not None:
            protocol, distance, normalize_features, max_rank, query_chunk_size = (
                config.protocol,
                config.distance,
                config.normalize_features,
                config.max_rank,
                config.query_chunk_size,
            )
            (
                junk_ids,
                exclude_same_id_same_camera,
                top_k_diagnostics,
                include_per_query,
            ) = (
                config.junk_ids,
                config.exclude_same_id_same_camera,
                config.top_k_diagnostics,
                config.include_per_query,
            )
        if max_rank < 1:
            raise InputValidationError("max_rank must be positive")
        if query_chunk_size is not None and query_chunk_size < 1:
            raise InputValidationError("query_chunk_size must be positive")
        if top_k_diagnostics < 0:
            raise InputValidationError("top_k_diagnostics cannot be negative")
        if config is not None and config.invalid_query_policy != "exclude":
            raise InputValidationError("invalid_query_policy must be 'exclude'")
        self.config = EvaluationConfig(
            protocol,
            distance,
            normalize_features,
            max_rank,
            query_chunk_size,
            tuple(junk_ids),
            exclude_same_id_same_camera,
            "exclude",
            top_k_diagnostics,
            include_per_query,
        )

    def evaluate(
        self,
        query_features=None,
        gallery_features=None,
        query_ids=None,
        gallery_ids=None,
        query_camids=None,
        gallery_camids=None,
        *,
        inputs: EvaluationInput | None = None,
        query_sample_ids=None,
        gallery_sample_ids=None,
    ) -> EvaluationResult:
        if inputs is None:
            inputs = EvaluationInput(
                query_features,
                gallery_features,
                query_ids,
                gallery_ids,
                query_camids,
                gallery_camids,
                query_sample_ids,
                gallery_sample_ids,
            )
        qf = validate_features(inputs.query_features, "query_features")
        gf = validate_features(inputs.gallery_features, "gallery_features")
        if qf.shape[1] != gf.shape[1]:
            raise InputValidationError("query and gallery feature dimensions differ")
        nq, ng = len(qf), len(gf)
        qids = validate_vector(inputs.query_ids, nq, "query_ids")
        gids = validate_vector(inputs.gallery_ids, ng, "gallery_ids")
        qcams = validate_vector(inputs.query_camids, nq, "query_camids")
        gcams = validate_vector(inputs.gallery_camids, ng, "gallery_camids")
        qs = (
            np.arange(nq, dtype=np.int64)
            if inputs.query_sample_ids is None
            else validate_vector(inputs.query_sample_ids, nq, "query_sample_ids")
        )
        gs = (
            np.arange(ng, dtype=np.int64)
            if inputs.gallery_sample_ids is None
            else validate_vector(inputs.gallery_sample_ids, ng, "gallery_sample_ids")
        )
        if len(set(map(str, qs))) != nq or len(set(map(str, gs))) != ng:
            raise InputValidationError("sample_ids must be unique")
        protocol = resolve_protocol(
            self.config.protocol,
            self.config.junk_ids,
            self.config.exclude_same_id_same_camera,
        )
        for left, right, name in (
            (qids, gids, "person IDs"),
            (qcams, gcams, "camera IDs"),
        ):
            if (left.dtype.kind in "iu") != (right.dtype.kind in "iu"):
                raise InputValidationError(
                    f"query/gallery {name} must have compatible types"
                )
            if (
                left.dtype.kind in "iu"
                and np.result_type(left.dtype, right.dtype).kind == "f"
            ):
                raise InputValidationError(
                    f"query/gallery {name} have unsafe signed/unsigned types; normalize to a common integer dtype"
                )
        protocol.validate(qids, gids, qcams, gcams)
        effective = self.config.as_dict()
        effective.update(
            protocol=protocol.name,
            junk_ids=list(protocol.junk_ids),
            exclude_same_id_same_camera=protocol.exclude_same_id_same_camera,
        )
        chunk = self.config.query_chunk_size or min(nq, 128)
        per: list[QueryResult] = []
        for start in range(0, nq, chunk):
            ds = distances(
                qf[start : start + chunk],
                gf,
                self.config.distance,
                self.config.normalize_features,
            )
            for local, row in enumerate(ds):
                i = start + local
                cand, posmask = protocol.mask(qids[i], qcams[i], gids, gcams)
                order = stable_order(row[cand])
                indices = np.flatnonzero(cand)[order]
                rel = posmask[indices]
                positives = int(rel.sum())
                excluded = int(ng - len(indices))
                top = []
                for match_rank, gi in enumerate(
                    indices[: self.config.top_k_diagnostics]
                    if self.config.include_per_query
                    else [],
                    1,
                ):
                    top.append(
                        {
                            "rank": match_rank,
                            "gallery_index": int(gi),
                            "gallery_sample_id": gs[gi].item()
                            if hasattr(gs[gi], "item")
                            else gs[gi],
                            "gallery_pid": gids[gi].item()
                            if hasattr(gids[gi], "item")
                            else gids[gi],
                            "gallery_camid": gcams[gi].item()
                            if hasattr(gcams[gi], "item")
                            else gcams[gi],
                            "distance": float(row[gi]),
                            "is_positive": bool(gids[gi] == qids[i]),
                        }
                    )
                sid = qs[i].item() if hasattr(qs[i], "item") else qs[i]
                pid = qids[i].item() if hasattr(qids[i], "item") else qids[i]
                cam = qcams[i].item() if hasattr(qcams[i], "item") else qcams[i]
                if positives == 0:
                    if any(pid == junk for junk in protocol.junk_ids):
                        reason = "junk_query_identity"
                    elif not len(indices):
                        reason = "empty_filtered_gallery"
                    elif not np.any(gids == qids[i]):
                        reason = "identity_absent"
                    else:
                        reason = "no_eligible_positive"
                    per.append(
                        QueryResult(
                            i,
                            sid,
                            pid,
                            cam,
                            False,
                            reason,
                            len(indices),
                            excluded,
                            0,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            top,
                        )
                    )
                    continue
                ranks = np.flatnonzero(rel) + 1
                first, last = int(ranks[0]), int(ranks[-1])
                hits = np.cumsum(rel)
                ap = float(np.sum(hits[ranks - 1] / ranks) / positives)
                inp = float(positives / last)
                per.append(
                    QueryResult(
                        i,
                        sid,
                        pid,
                        cam,
                        True,
                        None,
                        len(indices),
                        excluded,
                        positives,
                        first,
                        last,
                        ap,
                        inp,
                        int(first <= 1),
                        int(first <= 5) if ng >= 5 else None,
                        int(first <= 10) if ng >= 10 else None,
                        top,
                    )
                )
        valid = [x for x in per if x.valid]
        if not valid:
            raise EvaluationError(
                "all queries are invalid: no query has an eligible positive"
            )
        cmc = [
            float(
                np.mean(
                    [
                        q.first_positive_rank is not None and q.first_positive_rank <= r
                        for q in valid
                    ]
                )
            )
            if r <= ng
            else None
            for r in range(1, self.config.max_rank + 1)
        ]

        def rank(r):
            return (
                float(
                    np.mean(
                        [
                            q.first_positive_rank is not None
                            and q.first_positive_rank <= r
                            for q in valid
                        ]
                    )
                )
                if r <= ng
                else None
            )

        # Population fingerprints and feature fingerprints are kept separate:
        # changed embeddings are expected when comparing models, while the
        # metadata fingerprints establish whether the evaluation population is
        # paired and compatible.
        metadata = {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "ranking": "stable ascending distance, original gallery index ties",
            "protocol_version": protocol.version,
            "protocol_rules": protocol.rules(),
            "protocol_status": protocol.status,
            "metric_definitions": "multishot-first-match-ap-positive-mean-inp-last-positive-v1",
            "fingerprint_version": 2,
            "query_sample_ids_provided": inputs.query_sample_ids is not None,
            "gallery_sample_ids_provided": inputs.gallery_sample_ids is not None,
            "valid_query_fingerprint": array_fingerprint(
                np.array([q.query_index for q in valid], dtype=np.int64)
            ),
            "query_input_dtype": str(np.asarray(inputs.query_features).dtype),
            "gallery_input_dtype": str(np.asarray(inputs.gallery_features).dtype),
            "feature_dimension": int(qf.shape[1]),
            "feature_dtype": str(qf.dtype),
            "query_fingerprint": array_fingerprint(qids, qcams, qs),
            "gallery_fingerprint": array_fingerprint(gids, gcams, gs),
            "query_feature_fingerprint": array_fingerprint(qf),
            "gallery_feature_fingerprint": array_fingerprint(gf),
        }
        return EvaluationResult(
            cmc,
            rank(1),
            rank(5),
            rank(10),
            float(np.mean([q.average_precision for q in valid])),
            float(np.mean([q.inverse_negative_penalty for q in valid])),
            nq,
            len(valid),
            nq - len(valid),
            ng,
            per if self.config.include_per_query else [],
            protocol.name,
            effective,
            metadata,
        )

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from .exceptions import InputValidationError


@dataclass(frozen=True)
class Protocol:
    name: str
    version: str
    junk_ids: tuple[Any, ...]
    exclude_same_id_same_camera: bool = True
    status: str = "documented compatibility variant"

    def rules(self) -> dict[str, Any]:
        return {
            "junk_ids": list(self.junk_ids),
            "exclude_same_id_same_camera": self.exclude_same_id_same_camera,
            "invalid_query_policy": "exclude",
        }

    def validate(
        self, qids: np.ndarray, gids: np.ndarray, qcams: np.ndarray, gcams: np.ndarray
    ) -> None:
        if self.name == "custom":
            return
        if any(a.dtype.kind not in "iu" for a in (qids, gids, qcams, gcams)):
            raise InputValidationError(
                f"{self.name} requires integer IDs and zero-based camera IDs; use custom for renamed labels"
            )
        if self.name == "market1501" and any(
            np.any(a < -1) or np.any(a > 1501) for a in (qids, gids)
        ):
            raise InputValidationError(
                "market1501 requires original -1..1501 person IDs; use custom for renamed labels"
            )
        if self.name == "msmt17" and any(np.any(a < 0) for a in (qids, gids)):
            raise InputValidationError(
                "msmt17 has no defined negative junk identities; use custom for sentinels"
            )
        for a in (qcams, gcams):
            if np.any(a < 0) or np.any(a >= (6 if self.name == "market1501" else 15)):
                raise InputValidationError(
                    f"{self.name} requires zero-based camera IDs"
                )

    def mask(
        self, qpid: Any, qcam: Any, gpids: np.ndarray, gcams: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        candidate = np.ones(len(gpids), dtype=bool)
        for jid in self.junk_ids:
            candidate &= gpids != jid
        if self.exclude_same_id_same_camera:
            candidate &= ~((gpids == qpid) & (gcams == qcam))
        positive = candidate & (gpids == qpid)
        return candidate, positive


def resolve_protocol(
    name: str,
    junk_ids: tuple[Any, ...] | None = None,
    exclude_same_id_same_camera: bool = True,
) -> Protocol:
    n = name.lower()
    if n == "market1501":
        # Match the loader+rank pipeline: remove loader junk (-1), retain
        # background (0), then remove same-person/same-camera gallery entries.
        return Protocol(n, "market1501-torchreid-loader-rank-v2", (-1,), True)
    if n == "msmt17":
        return Protocol(
            n,
            "msmt17-fastreid-metadata-v2",
            (),
            True,
            "experimental: supplied splits, source-verified kernel",
        )
    if n == "custom":
        return Protocol(
            n,
            "custom-v1",
            tuple((-1,) if junk_ids is None else junk_ids),
            exclude_same_id_same_camera,
        )
    raise InputValidationError(f"unknown protocol: {name}")

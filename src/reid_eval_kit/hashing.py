from __future__ import annotations

import hashlib
import json

import numpy as np


def array_fingerprint(*arrays: np.ndarray) -> str:
    """SHA-256 of length-framed, little-endian arrays; never object pointers."""
    h = hashlib.sha256(b"reid-eval-kit-array-v2\0")
    for a in arrays:
        x = np.asarray(a)
        if x.dtype.hasobject:
            raise ValueError("object arrays cannot be reproducibly fingerprinted")
        x = np.ascontiguousarray(x, dtype=x.dtype.newbyteorder("<"))
        header = json.dumps(
            [x.dtype.str, list(x.shape)], separators=(",", ":")
        ).encode()
        h.update(len(header).to_bytes(8, "big"))
        h.update(header)
        h.update(x.nbytes.to_bytes(8, "big"))
        h.update(x.tobytes())
    return h.hexdigest()

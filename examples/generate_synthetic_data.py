from pathlib import Path
import numpy as np

rng = np.random.default_rng(42)
out = Path("synthetic")
out.mkdir(exist_ok=True)
q = rng.normal(size=(4, 8))
g = rng.normal(size=(12, 8))
np.savez(
    out / "query.npz",
    features=q,
    sample_ids=np.array([f"q{i}" for i in range(len(q))]),
    pids=np.array([0, 1, 2, 3]),
    camids=np.array([1, 1, 1, 1]),
)
np.savez(
    out / "gallery.npz",
    features=g,
    sample_ids=np.array([f"g{i}" for i in range(len(g))]),
    pids=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
    camids=np.array([2] * 12),
)
print(out)

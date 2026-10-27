# Local-file safety

NumPy files always use `allow_pickle=False`; YAML uses safe loading and all
configuration keys are validated. Inputs never execute code or load remote
resources. Manifest embedding paths must be relative and confined to the
manifest directory after symlink resolution. Images are never opened.

Output filenames are fixed, existing reports require explicit overwrite, and
symlink output files are rejected even with overwrite. CLI also guards paths
that would replace configuration or manifest inputs. Each file is written
atomically; a directory is not a multi-file transaction. Use one writer and a
fresh output directory. Filesystem mutation by another process during a run is
outside this toolkit's guarantees.

There is no decompression/allocation quota for large arrays. Apply OS resource
limits to hostile files. Reports contain metadata and fingerprints, never
embedding vectors or unnecessary absolute input paths. No security contact or
private reporting address is fabricated; use the repository maintainer's
established channel when available.

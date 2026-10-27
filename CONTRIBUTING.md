# Contributing

## Development environment

Use Python 3.11 or newer. From the repository root, create an isolated
environment with Python's built-in tools.

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Then install the editable package and development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pip check
```

The `dev` extra supplies pytest, coverage tooling, Ruff, mypy, build, Twine,
PyYAML, and YAML type stubs. Registry access is required unless these packages
are available in a local wheel directory. See [Getting started](docs/getting-started.md)
for offline installation and virtual-environment troubleshooting.

## Required checks

Run from the repository root with the virtual environment active:

```bash
ruff check .
ruff format --check .
mypy src/
python -m pytest -ra
python -m build
python -m twine check dist/*
```

Formatting changes can be applied with `ruff format .`. Build creates both
the wheel and source archive in `dist/`; it does not publish either artifact.

Keep the reference oracle independent from production helpers. Add hand-derived
and randomized regressions for semantic fixes. Protocol changes require source
provenance, explicit effective rules, and a new protocol version. Do not bundle
restricted datasets or quote synthetic results as model benchmark scores.

## Installed-wheel validation

After building, use a second virtual environment for the wheel. Install it
following the [wheel instructions](docs/getting-started.md#install-a-local-wheel),
then change to a directory outside the checkout and run:

```bash
python -I -c "import reid_eval_kit; print(reid_eval_kit.__version__)"
python -I -m reid_eval_kit --help
python -m pip check
```

Also run a synthetic evaluation from that directory. Generate input files with
`examples/generate_synthetic_data.py` using its absolute path, then run the
evaluate/inspect/compare commands in the README. Clear any custom `PYTHONPATH`
before testing and confirm imports resolve to the second environment's
`site-packages`, rather than the checkout. Editable-install tests alone do not
validate the wheel.

## Reproducibility and release evidence

Record the interpreter and installed dependency versions alongside each
validation run:

```bash
python --version
python -m pip freeze
```

Dependencies currently have lower bounds rather than a committed, complete
lockfile. Installing at a later date can resolve newer versions. An environment
snapshot helps reproduce a run but does not establish cross-platform locking.
The [release audit](docs/release-audit.md) records checks actually executed and
remaining conditions; configured commands are not proof they passed.

CI is configured for Python 3.11/3.12/3.13. Only executed versions count as
verification. No automatic publishing workflow is configured.

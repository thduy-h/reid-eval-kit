# Getting started

## Requirements

- Python 3.11 or newer, with `venv` and `pip` available.
- A local source checkout or a locally built wheel of `reid-eval-kit`.
- Access to a Python package index, or a local directory containing compatible
  dependency wheels.

The commands below use standard Python tooling. NumPy is the only mandatory
runtime dependency; YAML configuration additionally requires PyYAML. CPU-only
evaluation is supported. No training framework or dataset download is needed.

## Create a virtual environment

Open a terminal at the repository root. Confirm the Python version, then
create and activate a virtual environment.

Linux or macOS:

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -3 --version
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Once activated, use `python` for subsequent commands. Check which interpreter
is active:

```bash
python -c "import sys; print(sys.executable)"
python -m pip install --upgrade pip
```

The printed path should point inside `.venv`. Activation affects only the
current terminal; repeat the activation step in each new terminal.

## Install from source

Run at the repository root in the activated environment:

```bash
python -m pip install '.[yaml]'
python -m pip check
python -m reid_eval_kit --help
```

For NumPy-only installation, use `python -m pip install .`. JSON configuration
works without the YAML extra. Contributors should use
`python -m pip install -e '.[dev]'` instead; see [CONTRIBUTING](../CONTRIBUTING.md).

The package has not been published to PyPI. These commands install local source,
not a registry release named `reid-eval-kit`.

## Run the synthetic example

From the checkout root, create deterministic input files:

```bash
python examples/generate_synthetic_data.py
```

The following commands are each on one line so they work in Bash and PowerShell:

```bash
python -m reid_eval_kit evaluate --query synthetic/query.npz --gallery synthetic/gallery.npz --protocol custom --distance cosine --normalize --output results/example
python -m reid_eval_kit inspect results/example/metrics.json
python -m reid_eval_kit evaluate --query synthetic/query.npz --gallery synthetic/gallery.npz --protocol custom --distance cosine --normalize --query-chunk-size 1 --output results/repeat
python -m reid_eval_kit compare results/example/metrics.json results/repeat/metrics.json
```

The comparison should report compatible runs with zero metric differences.
These are synthetic results, not research dataset benchmark scores. Use fresh
output directories for another run, or pass `--overwrite` explicitly.

The installed `reideval` command is equivalent to `python -m reid_eval_kit`.
See [input formats](input-formats.md) before evaluating your own embeddings and
[protocol provenance](protocols.md) before choosing a dataset protocol.

## Install a local wheel

To build distributions, first install the development extra and run:

```bash
python -m build
```

Create and activate a second environment from the checkout root. If another
environment is active, leave it first with `deactivate`.

Linux or macOS:

```bash
python3 -m venv .venv-wheel
source .venv-wheel/bin/activate
```

Windows PowerShell:

```powershell
py -3 -m venv .venv-wheel
.\.venv-wheel\Scripts\Activate.ps1
```

Then install the wheel using its path:

```bash
python -m pip install 'dist/reid_eval_kit-0.1.0-py3-none-any.whl[yaml]'
python -m pip check
python -m reid_eval_kit --help
```

Omit `[yaml]` for the NumPy-only installation. A wheel installation does not
include the checkout's examples or documentation; retain the source checkout
if you want to run the example script.

## Offline installation

On a connected machine with the same target Python version, operating system,
and architecture, download the built wheel and its runtime dependencies into
a directory:

```bash
python -m pip download --dest wheelhouse --only-binary=:all: 'dist/reid_eval_kit-0.1.0-py3-none-any.whl[yaml]'
```

Transfer `wheelhouse/` to the target machine. In its activated virtual
environment, install without accessing a package index:

```bash
python -m pip install --no-index --find-links wheelhouse 'reid-eval-kit[yaml]==0.1.0'
python -m pip check
python -m reid_eval_kit --help
```

These instructions cover runtime dependencies. Offline source builds and
development installs also require their build and development dependencies;
a runtime wheel directory alone is insufficient. Downloading NumPy wheels for
one platform does not make them usable on another platform.

## Troubleshooting

- **`venv` or `ensurepip` is unavailable:** install your operating system's
  Python virtual-environment support. Debian/Ubuntu commonly provides it as
  `python3-venv`. Then recreate the environment.
- **PowerShell blocks activation:** activation is optional. Run the environment
  interpreter directly, for example
  `.\.venv\Scripts\python.exe -m pip install '.[yaml]'` and
  `.\.venv\Scripts\python.exe -m reid_eval_kit --help`.
- **`reideval` is not found:** activate the intended environment, or use
  `python -m reid_eval_kit` with that environment's interpreter.
- **No compatible NumPy wheel:** check the Python version, OS, and architecture.
  A source build requires additional compiler tooling; the offline binary-only
  download command deliberately fails rather than assuming it is available.
- **NumPy cannot load a shared library:** check the binary package's system
  runtime requirements or use your platform's compatible NumPy distribution.
  Reinstalling the toolkit does not supply system shared libraries.
- **Output already exists:** choose another output directory or use
  `--overwrite` when replacement is intended.

For reproducibility, save `python --version` and `python -m pip freeze` with
your experiment. See [reproducibility](reproducibility.md) for fingerprints,
ranking rules, and numerical limits.

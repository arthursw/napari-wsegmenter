# napari-wsegmenter

[![License MIT](https://img.shields.io/pypi/l/napari-wsegmenter.svg?color=green)](https://github.com/arthursw/napari-wsegmenter/raw/main/LICENSE)
[![npe2](https://img.shields.io/badge/plugin-npe2-blue?link=https://napari.org/stable/plugins/index.html)](https://napari.org/stable/plugins/index.html)

Segment images with Cellpose, StarDist, or SAM 2 while keeping their dependencies out of the environment that runs napari.

This branch is the integration example for napari-managed plugin environments.
It requires the corresponding napari and npe2 feature branches and Wetlands 2.3.3 or later.
The plugin itself does not import or depend on Wetlands.
Wetlands is a private execution backend behind napari-owned task, progress, failure, and lifecycle APIs.

## How dependency isolation works

The host code that napari imports contains only the Qt widgets and napari integration.
Its runtime requirements are exactly napari, NumPy, and QtPy, which are the host packages it imports and which the running napari installation already supplies.
Importing the plugin does not import Cellpose, TensorFlow, StarDist, SAM, PyTorch, or Wetlands.

The plugin manifest declares a separate managed environment for each segmenter.
It also associates each worker command with its environment:

- `napari-wsegmenter.cellpose_worker` runs with Cellpose;
- `napari-wsegmenter.stardist_worker` runs with TensorFlow and StarDist;
- `napari-wsegmenter.sam_worker` runs with SAM 2 and PyTorch.

The worker code is a single module in the two-file embedded project at `src/napari_wsegmenter/worker`.
The outer `napari-wsegmenter` wheel ships that project as package data, and napari installs it into each managed environment without publishing a second package.
Worker entry points are qualified Python targets.
They accept a NumPy image and a plain parameter dictionary, then return a NumPy labels array.
They do not import napari GUI APIs.

The widget calls `napari.plugins.execute_worker_command`, shows compact status and progress for the active request, exposes cancellation, presents failures through napari notifications, and adds returned labels in the main napari process.
Napari owns startup environment setup, worker lifecycle, transport, and shutdown.

## Installation and first run

This is a coordinated pre-release integration branch.
Install `wetlands>=2.3.3,<2.4`, plus the coordinated npe2 and napari feature checkouts, into one development environment before installing this plugin; do not infer a future napari release number from the currently unbounded `napari` requirement.
Before publishing WSegmenter, replace that requirement with a lower bound on the first released napari version that provides managed plugin environments.

Then install this plugin into that environment:

```sh
pip install -e .
```

Installing, updating, or uninstalling the plugin requires restarting napari before the new host code and manifest are used.
On restart, napari automatically installs or updates all three declared environments before worker commands become available.
Unchanged environments are reused without showing the setup dialog.
Large scientific environments may take several minutes to install, and model frameworks may also download assets during execution.
If setup for a segmenter fails or is skipped, its widget reports that the environment is unavailable and asks the user to restart napari to retry setup.
Opening a segmenter widget and running a command never installs, removes, or updates an environment.

The first command for an installed environment starts its worker lazily.
Later commands reuse the warm worker until it is stopped from napari's **Plugin Environments** window or napari exits.
Each widget reports command progress and can cancel its own queued or running command.

The same restart rule applies when installing the plugin directly with `pip`: restart napari so discovery sees the changed distribution and reconciles its declared environments.

The three segmenters are intentionally isolated from one another.
Their framework versions do not alter napari's packages or constrain the dependencies of another plugin environment.

The SAM environment installs Meta's official SAM 2 source at the immutable Git commit declared in `napari.yaml`.
Meta does not publish an official SAM 2 distribution on PyPI, so the recipe deliberately does not use the unrelated third-party `sam2` project from PyPI.
The worker converts 2D grayscale images to RGB before calling SAM 2.
Meta documents Linux as its supported platform.
macOS arm64 CPU execution was manually exercised during development, but it is not covered by this repository's automated test matrix and remains outside Meta's upstream support statement.

## Packaging contract

Plugin GUI and napari integration code runs in the napari process.
The main plugin distribution may require only napari and packages in napari's direct base requirements for the current platform, and every version constraint must accept the version already installed with napari.
Every other runtime dependency, and the code that imports it, belongs in `contributions.environments` and worker code rather than the main distribution.
Napari-managed installation validates the exact plugin wheel against this rule and rejects a nonconforming wheel before installing it without dependency resolution.
The host plugin requires Python 3.11 or later; its managed worker environments select Python 3.10 independently to match their scientific frameworks.
The manifest is authoritative for worker runtime dependencies, including NumPy.
The embedded worker project's dependency list is deliberately empty; its `pyproject.toml` exists only to make the adjacent `napari_wsegmenter_worker.py` module an installable qualified target.

The embedded project must be included in both the source distribution and wheel because `contributions.worker_package` is resolved relative to the installed manifest.
Napari fingerprints and immutably stages its complete contents, so worker code or resource changes rebuild the environment without relying on the embedded project's distribution version.
Plugin authors expose installed `module:callable` targets.
Filesystem path execution and backend transport are not part of the napari plugin API.

Existing plugins installed with `pip`, Conda, or another external tool remain discoverable, but those installation flows use normal dependency resolution and are outside napari's isolation guarantee.
A plugin must follow the host dependency contract before napari's managed installer can accept it.

Managed environments isolate Python dependencies, but they are not security sandboxes.
Worker code is trusted plugin code and retains the user's filesystem, network, process, GPU, and credential access.

## Usage

Select an image layer, open the Cellpose, StarDist, or SAM dock widget, choose parameters, and click Run.
Returned labels are added as a napari Labels layer.
Each widget keeps the plugin-specific interface compact: it displays command status and progress and provides Run and Cancel controls.
Startup setup and worker lifecycle logs are centralized in napari's **Plugin Environments** window instead of being duplicated in each plugin widget.
That window can stop an idle warm worker to release memory, but environments are otherwise immutable during a napari session.

For local development, launch napari with:

```sh
uv run python launch_napari.py
```

Open the segmenter you want from **Plugins > WSegmenter**.
The menu action owns each dock widget's creation and then toggles the same widget rather than creating duplicates.

## Development and tests

Install the testing dependencies and run the documented test suite:

```sh
uv sync --extra testing
uv run tox run
```

Build and validate the release artifact against the running napari environment:

```sh
uv run python -m build
uv run npe2 validate --host-dependencies dist/napari_wsegmenter-*.whl
```

The unit tests replace heavy frameworks and the napari runtime task with fakes.
They verify lazy imports, ordinary NumPy inputs and outputs, progress, cancellation, failures, result-layer creation, npe2 manifest parsing, authoritative wheel metadata, and embedded worker contents without installing multi-gigabyte environments.
A real end-to-end smoke test should be run from the matching napari and npe2 feature branches before release.

## License

Distributed under the terms of the [MIT license](LICENSE).

## Issues

Please [file an issue](https://github.com/arthursw/napari-wsegmenter/issues) with a detailed description of any problem.

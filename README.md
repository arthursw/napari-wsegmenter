# napari-wsegmenter

[![License MIT](https://img.shields.io/pypi/l/napari-wsegmenter.svg?color=green)](https://github.com/arthursw/napari-wsegmenter/raw/main/LICENSE)
[![npe2](https://img.shields.io/badge/plugin-npe2-blue?link=https://napari.org/stable/plugins/index.html)](https://napari.org/stable/plugins/index.html)

Segment images with Cellpose, StarDist, or SAM 2 while keeping their dependencies out of the environment that runs napari.

This branch is the integration example for napari-managed plugin environments.
It requires the corresponding napari and npe2 feature branches and, until Wetlands 2 is published, a local editable installation of Wetlands 2 in the napari development environment.
The plugin itself does not import or depend on Wetlands.
Wetlands is a private execution backend behind napari-owned task, progress, failure, and lifecycle APIs.

## How dependency isolation works

The installed `napari-wsegmenter` host package contains only the Qt widgets and napari integration.
It depends on NumPy and QtPy, and importing it does not import Cellpose, TensorFlow, StarDist, SAM, PyTorch, or Wetlands.

The plugin manifest declares a separate managed environment for each segmenter.
It also associates each worker command with its environment:

- `napari-wsegmenter.cellpose_worker` runs with Cellpose;
- `napari-wsegmenter.stardist_worker` runs with TensorFlow and StarDist;
- `napari-wsegmenter.sam_worker` runs with SAM 2 and PyTorch.

The small worker distribution in `src/napari_wsegmenter/worker-package` is shipped as plugin package data and installed into each managed environment.
Worker entry points are qualified Python targets.
They accept a NumPy image and a plain parameter dictionary, then return a NumPy labels array.
They do not import napari GUI APIs.

The widget calls `napari.plugins.execute_worker_command`, reports preparation and execution updates, exposes cancellation, presents failures through napari notifications, and adds returned labels in the main napari process.
Napari owns provisioning, worker reuse, transport, and shutdown.

## Installation and first run

Install this plugin into a napari environment that contains the managed-environment feature:

```sh
pip install -e .
```

Opening a segmenter widget is side-effect-free.
The first Run provisions that segmenter's environment and may take several minutes while packages and model assets are downloaded.
The widget displays provisioning and execution progress and can request cancellation.
Later runs reuse the provisioned environment and warm worker while its declared recipe is unchanged.
Changing the recipe causes napari to build a new environment generation.

The three segmenters are intentionally isolated from one another.
Their framework versions do not alter napari's packages or constrain the dependencies of another plugin environment.

The SAM environment installs Meta's official SAM 2 source at the immutable Git commit declared in `napari.yaml`.
Meta does not publish an official SAM 2 distribution on PyPI, so the recipe deliberately does not use the unrelated third-party `sam2` project from PyPI.
The worker converts 2D grayscale images to RGB before calling SAM 2.
Meta documents Linux as its supported platform; macOS arm64 CPU execution is validated here as an integration example but remains outside Meta's upstream support statement.

## Packaging contract

Plugin GUI and napari integration code must remain lightweight enough to install in the napari environment.
Dependencies needed only by worker functionality belong in `contributions.environments`, not in the host package dependencies.
The worker distribution must be included in both the source distribution and wheel because its `local_packages` path is resolved relative to the installed manifest.
Its distribution version must change whenever worker code changes so package-build caches cannot reuse an older worker artifact.

Existing napari plugins continue to run in the host process unless they opt into managed worker commands.
Isolation can only be guaranteed for dependencies installed through napari-managed environments; users can still manually install conflicting packages into the napari environment.

Managed environments isolate Python dependencies, but they are not security sandboxes.
Worker code is trusted plugin code and retains the user's filesystem, network, process, GPU, and credential access.

## Usage

Select an image layer, open one of the Cellpose, StarDist, or SAM dock widgets, choose parameters, and click Run.
Returned labels are added as a napari Labels layer.

For local development, launch all three widgets with:

```sh
uv run python launch_napari.py
```

## Development and tests

Install the testing dependencies and run the documented test suite:

```sh
uv sync --extra testing
uv run tox run
```

The unit tests replace heavy frameworks and the napari runtime task with fakes.
They verify lazy imports, ordinary NumPy inputs and outputs, progress, cancellation, failures, and result-layer creation without provisioning multi-gigabyte environments.
A real end-to-end smoke test should be run from the matching napari and npe2 feature branches before release.

## License

Distributed under the terms of the [MIT license](LICENSE).

## Issues

Please [file an issue](https://github.com/arthursw/napari-wsegmenter/issues) with a detailed description of any problem.

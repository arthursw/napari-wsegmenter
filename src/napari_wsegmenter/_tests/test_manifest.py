import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from zipfile import ZipFile

import pytest
import yaml
from npe2 import (
    HostDependencyPolicy,
    PackageMetadata,
    PluginManifest,
    get_manifest_from_wheel,
    validate_host_dependencies,
)
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

PACKAGE_ROOT = Path(__file__).parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
MANIFEST_PATH = PACKAGE_ROOT / "napari.yaml"
WORKER_ROOT = PACKAGE_ROOT / "worker"


def _manifest():
    return yaml.safe_load(MANIFEST_PATH.read_text())


def _environment(environment_id):
    manifest = _manifest()
    environments = manifest["contributions"]["environments"]
    return next(
        environment
        for environment in environments
        if environment["id"] == environment_id
    )


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    temporary_root = tmp_path_factory.mktemp("wheel-build")
    source_root = temporary_root / "source"
    output_root = temporary_root / "dist"
    shutil.copytree(
        PROJECT_ROOT,
        source_root,
        ignore=shutil.ignore_patterns(
            ".git",
            ".tox",
            ".venv",
            "__pycache__",
            "*.egg-info",
            "build",
            "dist",
        ),
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(output_root),
            str(source_root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    wheels = list(output_root.glob("*.whl"))
    assert len(wheels) == 1
    return wheels[0]


def _active_base_requirements(metadata: PackageMetadata) -> set[str]:
    environment = {**default_environment(), "extra": ""}
    requirements = (
        Requirement(value) for value in metadata.requires_dist or ()
    )
    return {
        canonicalize_name(requirement.name)
        for requirement in requirements
        if requirement.marker is None
        or requirement.marker.evaluate(environment=environment)
    }


def test_manifest_validates_with_npe2():
    manifest = PluginManifest.from_file(MANIFEST_PATH)

    assert manifest.name == "napari-wsegmenter"


def test_built_wheel_has_valid_host_dependencies(built_wheel: Path):
    manifest = get_manifest_from_wheel(str(built_wheel))
    assert manifest.package_metadata is not None

    validate_host_dependencies(
        manifest.package_metadata,
        HostDependencyPolicy.from_environment(),
    )
    assert _active_base_requirements(manifest.package_metadata) == {
        "napari",
        "numpy",
        "qtpy",
    }

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "npe2",
            "validate",
            "--host-dependencies",
            str(built_wheel),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Host dependencies" in completed.stdout


def test_built_wheel_contains_minimal_worker_project(built_wheel: Path):
    with ZipFile(built_wheel) as wheel:
        names = set(wheel.namelist())
        worker_project = tomllib.loads(
            wheel.read("napari_wsegmenter/worker/pyproject.toml").decode()
        )

    assert "napari_wsegmenter/worker/napari_wsegmenter_worker.py" in names
    assert worker_project["project"]["dependencies"] == []
    assert worker_project["tool"]["setuptools"]["py-modules"] == [
        "napari_wsegmenter_worker"
    ]


def test_environments_use_flat_embedded_worker_project():
    environments = _manifest()["contributions"]["environments"]

    assert {environment["display_name"] for environment in environments} == {
        "Cellpose",
        "SAM 2",
        "StarDist",
        "Threshold NumPy 1.26",
        "Threshold NumPy 2.2",
    }
    for environment in environments:
        assert environment["provision"] == "on_demand"
        assert environment["local_packages"] == [{"path": "worker"}]


def test_environment_manifest_owns_worker_dependencies():
    cellpose = _environment("napari-wsegmenter.cellpose")
    stardist = _environment("napari-wsegmenter.stardist")
    sam = _environment("napari-wsegmenter.sam")
    numpy1 = _environment("napari-wsegmenter.threshold_numpy1")
    numpy2 = _environment("napari-wsegmenter.threshold_numpy2")

    assert "numpy" in cellpose["conda"]
    assert "numpy>=1.23.5,<2" in stardist["pypi"]
    assert "numpy>=1.24.4" in sam["pypi"]
    assert numpy1["conda"] == ["numpy==1.26.4"]
    assert numpy2["conda"] == ["numpy==2.2.6"]

    worker_project = (WORKER_ROOT / "pyproject.toml").read_text()
    assert "dependencies = []" in worker_project
    assert 'py-modules = ["napari_wsegmenter_worker"]' in worker_project


def test_worker_commands_remain_qualified_import_targets():
    commands = _manifest()["contributions"]["commands"]
    worker_commands = [
        command for command in commands if "environment" in command
    ]

    assert {command["python_name"] for command in worker_commands} == {
        "napari_wsegmenter_worker:segment_cellpose",
        "napari_wsegmenter_worker:segment_sam",
        "napari_wsegmenter_worker:segment_stardist",
        "napari_wsegmenter_worker:segment_threshold",
    }
    assert all("path" not in command for command in worker_commands)


def test_worker_project_has_only_the_required_source_files():
    assert {path.name for path in WORKER_ROOT.iterdir() if path.is_file()} == {
        "napari_wsegmenter_worker.py",
        "pyproject.toml",
    }


def test_stardist_environment_avoids_removed_pkg_resources():
    stardist = _environment("napari-wsegmenter.stardist")

    assert "stardist==0.9.2" in stardist["pypi"]
    assert "csbdeep==0.8.2" in stardist["pypi"]


def test_sam_environment_uses_pinned_official_source():
    sam = _environment("napari-wsegmenter.sam")

    assert (
        "sam-2 @ git+https://github.com/facebookresearch/sam2.git"
        "@2b90b9f5ceec907a1c18123530e92e794ad901a4"
    ) in sam["pypi"]
    assert not any(
        requirement.startswith("sam2==") for requirement in sam["pypi"]
    )

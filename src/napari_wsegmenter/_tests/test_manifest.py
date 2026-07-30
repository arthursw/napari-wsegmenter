from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).parents[1]
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


def test_environments_use_flat_embedded_worker_project():
    environments = _manifest()["contributions"]["environments"]

    assert {environment["display_name"] for environment in environments} == {
        "Cellpose",
        "SAM 2",
        "StarDist",
    }
    for environment in environments:
        assert environment["provision"] == "on_demand"
        assert environment["local_packages"] == [{"path": "worker"}]


def test_environment_manifest_owns_worker_dependencies():
    cellpose = _environment("napari-wsegmenter.cellpose")
    stardist = _environment("napari-wsegmenter.stardist")
    sam = _environment("napari-wsegmenter.sam")

    assert "numpy" in cellpose["conda"]
    assert "numpy>=1.23.5,<2" in stardist["pypi"]
    assert "numpy>=1.24.4" in sam["pypi"]

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

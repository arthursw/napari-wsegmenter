from pathlib import Path

import yaml


def _environment(environment_id):
    manifest_path = Path(__file__).parents[1] / "napari.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    environments = manifest["contributions"]["environments"]
    return next(
        environment
        for environment in environments
        if environment["id"] == environment_id
    )


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

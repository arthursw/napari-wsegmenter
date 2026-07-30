from pathlib import Path

import yaml


def test_stardist_environment_avoids_removed_pkg_resources():
    manifest_path = Path(__file__).parents[1] / "napari.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    environments = manifest["contributions"]["environments"]
    stardist = next(
        environment
        for environment in environments
        if environment["id"] == "napari-wsegmenter.stardist"
    )

    assert "stardist==0.9.2" in stardist["pypi"]
    assert "csbdeep==0.8.2" in stardist["pypi"]

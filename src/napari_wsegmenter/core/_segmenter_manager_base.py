import subprocess
import tempfile
from pathlib import Path
from typing import cast

import numpy as np
from napari.qt.threading import WorkerBase, thread_worker
from wetlands.environment import Environment
from wetlands.environment_manager import EnvironmentManager
from wetlands.external_environment import ExternalEnvironment

WETLANDS_INSTALL_DIR = Path.home() / ".local" / "share" / "wetlands"
WETLANDS_INSTALL_DIR.mkdir(parents=True, exist_ok=True)
PYTHON_VERSION = "3.10"


@thread_worker
def log_output(process: subprocess.Popen) -> None:
    if process.stdout is None:
        return
    for line in iter(process.stdout.readline, ""):
        print(line.strip())


class SegmenterManagerBase:

    config = {
        "Cellpose": {
            "name": "cellpose",
            "dependencies": {
                "python": PYTHON_VERSION,
                "conda": ["cellpose==3.1.0"],
            },
            "segmenter_script_name": "_cellpose.py",
            "default_parameters": {
                "model_type": "cyto3",
                "use_gpu": False,
                "diameter": 30.0,
            },
        },
        "StarDist": {
            "name": "stardist",
            "dependencies": {
                "python": PYTHON_VERSION,
                "pip": [
                    "tensorflow==2.16.1",
                    "csbdeep==0.8.1",
                    "stardist==0.9.1",
                ],
                "conda": [
                    {
                        "name": "nvidia::cudatoolkit=11.0.*",
                        "platforms": ["win-64", "linux-64"],
                        "optional": True,
                    },
                    {
                        "name": "nvidia::cudnn=8.0.*",
                        "platforms": ["win-64", "linux-64"],
                        "optional": True,
                    },
                ],
            },
            "segmenter_script_name": "_stardist.py",
            "default_parameters": {"model_name": "2D_versatile_fluo"},
        },
        "SAM": {
            "name": "sam",
            "dependencies": {
                "python": PYTHON_VERSION,
                "conda": ["sam2==1.1.0", "huggingface_hub==0.29.2"],
            },
            "segmenter_script_name": "_sam.py",
            "default_parameters": {
                "use_gpu": False,
                "points_per_side": 32,
                "pred_iou_thresh": 0.88,
                "stability_score_thresh": 0.95,
            },
        },
    }

    _environment_manager: EnvironmentManager | None = None
    _environment: Environment | None = None
    _segmenters_path = (
        Path(__file__).resolve().parent / "segmenters" / "shared_memory"
    )

    def _initialize_environment(self, name: str):
        config = self.config[name]
        if self._environment_manager is None:
            self._environment_manager = EnvironmentManager(
                str(WETLANDS_INSTALL_DIR / "micromamba")
            )
        self._environment = self._environment_manager.create(
            name, config["dependencies"]
        )
        launched = self._environment.launched()
        if not launched:
            self._environment.launch()
        segmenter_module = self._environment.importModule(
            str(self._segmenters_path / config["segmenter_script_name"])
        )
        if not launched:
            worker = cast(
                WorkerBase,
                log_output(
                    cast(ExternalEnvironment, self._environment).process
                ),
            )
            worker.start()
        return segmenter_module

    def perform_segmentation(self, image: np.ndarray, segmenter: str):
        segmenter_module = self._initialize_environment(segmenter)

        with tempfile.TemporaryDirectory() as tempdir:
            input_path = Path(tempdir) / "image.npy"
            output_path = Path(tempdir) / "segmentation.npy"
            np.save(input_path, cast(np.ndarray, image.data))
            segmenter_module.segment(
                input_path,
                output_path,
                self.config[segmenter]["default_parameters"],
            )
            return np.load(output_path)

    def exit_environment(self):
        if self._environment is None:
            return
        self._environment.exit()
        self._environment = None

    def exit(self):
        self.exit_environment()


if __name__ == "__main__":
    segmenter_manager = SegmenterManagerBase()
    result = cast(
        np.ndarray,
        segmenter_manager.perform_segmentation(
            np.random.random((100, 100)), "StarDist"
        ),
    )
    print(result.shape)
    segmenter_manager.exit()

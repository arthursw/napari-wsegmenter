import subprocess
from multiprocessing import shared_memory
from pathlib import Path
from typing import cast

import numpy as np
from napari.qt.threading import WorkerBase, thread_worker
from napari.viewer import Viewer
from qtpy.QtWidgets import QComboBox, QLabel, QPushButton, QVBoxLayout, QWidget
from wetlands.environment_manager import EnvironmentManager
from wetlands.external_environment import ExternalEnvironment

from napari_wsegmenter.core._memory_manager import (
    create_shared_array,
    release_shared_memory,
    share_array,
    wrap,
)

WETLANDS_INSTALL_DIR = Path.home() / ".local" / "share" / "wetlands"
WETLANDS_INSTALL_DIR.mkdir(parents=True, exist_ok=True)
PYTHON_VERSION = "3.10"
SEGMENTERS_PATH = (
    Path(__file__).resolve().parent / "core" / "segmenters" / "shared_memory"
)


@thread_worker
def log_output(process: subprocess.Popen) -> None:
    if process.stdout is None:
        return
    for line in iter(process.stdout.readline, ""):
        print(line.strip())


class ImageSegmenter(QWidget):

    config = {
        "Cellpose": {
            "name": "cellpose",
            "dependencies": {
                "python": PYTHON_VERSION,
                "conda": ["cellpose==3.1.0"],
            },
            "segmenter_script_name": SEGMENTERS_PATH / "_cellpose.py",
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
            "segmenter_script_name": SEGMENTERS_PATH / "_stardist.py",
            "default_parameters": {"model_name": "2D_versatile_fluo"},
        },
        "SAM": {
            "name": "sam",
            "dependencies": {
                "python": PYTHON_VERSION,
                "conda": ["sam2==1.1.0", "huggingface_hub==0.29.2"],
            },
            "segmenter_script_name": SEGMENTERS_PATH / "_sam.py",
            "default_parameters": {
                "use_gpu": False,
                "points_per_side": 32,
                "pred_iou_thresh": 0.88,
                "stability_score_thresh": 0.95,
            },
        },
    }

    _environment_manager: EnvironmentManager | None = None
    _shared_image: np.ndarray | None = None
    _shm_image: shared_memory.SharedMemory | None = None
    _shared_segmentation: np.ndarray | None = None
    _shm_segmentation: shared_memory.SharedMemory | None = None

    def __init__(self, viewer: Viewer):
        super().__init__()
        self._viewer = viewer

        # Create layout
        layout = QVBoxLayout()

        # Image layer combo
        self._image_layer_label = QLabel("Image")
        self._image_layer_combo = QComboBox()
        layout.addWidget(self._image_layer_label)
        layout.addWidget(self._image_layer_combo)

        # Segmenter combo
        self._segmenter_label = QLabel("Segmenter")
        self._segmenter_combo = QComboBox()
        self._segmenter_combo.addItems(["StarDist", "Cellpose", "SAM"])
        layout.addWidget(self._segmenter_label)
        layout.addWidget(self._segmenter_combo)

        # Run button
        self._run_button = QPushButton("Segment")
        self._run_button.clicked.connect(self.segment)
        layout.addWidget(self._run_button)

        self.setLayout(layout)

        # Populate image combo when viewer updates
        self._viewer.layers.events.inserted.connect(self._update_image_layers)
        self._viewer.layers.events.removed.connect(self._update_image_layers)
        self._update_image_layers()

    def _update_image_layers(self, event=None):
        self._image_layer_combo.clear()
        image_layers = [
            layer.name
            for layer in self._viewer.layers
            if layer.__class__.__name__ == "Image"
        ]
        self._image_layer_combo.addItems(image_layers)

    def _initialize_environment(self, name: str):
        config = self.config[name]
        if self._environment_manager is None:
            self._environment_manager = EnvironmentManager(
                str(WETLANDS_INSTALL_DIR / "micromamba")
            )
        environment = self._environment_manager.create(
            name, config["dependencies"]
        )
        launched = environment.launched()
        if not launched:
            environment.launch()
        segmenter_module = environment.importModule(
            str(config["segmenter_script_name"])
        )
        if not launched:
            worker = cast(
                WorkerBase,
                log_output(cast(ExternalEnvironment, environment).process),
            )
            worker.start()
        return segmenter_module

    def _initialize_shared_memory(self, image: np.ndarray):
        if (
            self._shared_image is not None
            and self._shm_image is not None
            and self._shared_segmentation is not None
            and self._shm_segmentation is not None
        ):
            if (
                self._shared_image.dtype == image.dtype
                and self._shared_image.shape == image.shape
            ):
                return
            else:
                self.release_shared_memory()
        self._shared_image, self._shm_image = share_array(image)
        _shared_image, _shm_image = share_array(image)
        self._shared_segmentation, self._shm_segmentation = (
            create_shared_array(image.shape, dtype="uint8")
        )

    def segment(self):
        image_name = self._image_layer_combo.currentText()
        segmenter = self._segmenter_combo.currentText()

        if image_name not in self._viewer.layers:
            return

        print(f"Running '{segmenter}' on image layer '{image_name}'")

        image_layer = self._viewer.layers[image_name]

        segmentation = self._perform_segmentation(image_layer.data, segmenter)

        name = image_layer.name + "_segmented"
        if name in self._viewer.layers:
            self._viewer.layers[name].data = segmentation
        else:
            self._viewer.add_labels(segmentation, name=name)

    def _perform_segmentation(self, image: np.ndarray, segmenter: str):
        segmenter_module = self._initialize_environment(segmenter)
        self._initialize_shared_memory(image)
        if self._shared_image is None or self._shared_segmentation is None:
            return
        if self._shm_image is None or self._shm_segmentation is None:
            return
        segmenter_module.segment(
            wrap(self._shared_image, self._shm_image),
            wrap(self._shared_segmentation, self._shm_segmentation),
            self.config[segmenter]["default_parameters"],
        )
        return self._shared_segmentation

    def closeEvent(self, a0):
        self._release_shared_memory()
        super().closeEvent(a0)

    def _release_shared_memory(self):
        release_shared_memory(self._shm_image)
        release_shared_memory(self._shm_segmentation)

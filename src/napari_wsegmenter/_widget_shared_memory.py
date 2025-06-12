import subprocess
from multiprocessing import shared_memory
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from magicgui import magic_factory
from napari.qt.threading import WorkerBase, thread_worker
from wetlands.environment_manager import EnvironmentManager

from napari_wsegmenter._memory_manager import (
    create_shared_array,
    share_array,
    wrap,
)

if TYPE_CHECKING:
    import napari
    import napari.types

WETLANDS_INSTALL_DIR = Path.home() / ".local" / "share" / "wetlands"
WETLANDS_INSTALL_DIR.mkdir(parents=True, exist_ok=True)
PYTHON_VERSION = "python=3.10"

SEGMENTERS_PATH = (
    Path(__file__).resolve().parent / "segmenters" / "shared_memory"
)

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
            "pip": ["tensorflow==2.16.1", "csbdeep==0.8.1", "stardist==0.9.1"],
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


class Global:
    shared_image: np.ndarray
    shm_image: shared_memory.SharedMemory
    shared_segmentation: np.ndarray
    shm_segmentation: shared_memory.SharedMemory
    environment_manager: EnvironmentManager


@thread_worker
def log_output(process: subprocess.Popen) -> None:
    if process.stdout is None:
        return
    for line in iter(process.stdout.readline, ""):
        print(line.strip())


def initialize_environment(name: str):
    if Global.environment_manager is None:
        Global.environment_manager = EnvironmentManager(
            str(WETLANDS_INSTALL_DIR / "pixi")
        )
    env = Global.environment_manager.create(name, config[name]["dependencies"])
    launched = env.launched()
    if not launched:
        env.launch()
    segmenter_module = env.importModule(
        str(config[name]["segmenter_script_name"])
    )
    if not launched:
        worker = cast(WorkerBase, log_output(segmenter_module.process))
        worker.start()
    return segmenter_module


def initialize_shared_memory(img):
    if (
        Global.shared_image is not None
        and Global.shm_image is not None
        and Global.shared_segmentation is not None
        and Global.shm_segmentation is not None
    ):
        if (
            Global.shared_image.dtype == img.dtype
            and Global.shared_image.shape == img.shape
        ):
            return
        else:
            release_shared_memory()
    Global.shared_image, Global.shm_image = share_array(img)
    Global.shared_segmentation, Global.shm_segmentation = create_shared_array(
        img.shape, dtype="uint8"
    )


def release_shared_memory():
    if Global.shm_image is None:
        return
    Global.shm_image.close()
    Global.shm_image.unlink()
    if Global.shm_segmentation is None:
        return
    Global.shm_segmentation.close()
    Global.shm_segmentation.unlink()


def _widget_initializer(widget):
    widget.root_native_widget.closeEvent().connect(release_shared_memory)


@magic_factory(
    segmenter={"choices": ["stardist", "cellpose", "sam"]},
    widget_init=_widget_initializer,
)
def segment_widget_shared_memory(
    img: "napari.types.ImageData",
    segmenter: "str",
) -> "napari.types.LabelsData":
    segmenter_module = initialize_environment(segmenter)
    initialize_shared_memory(img)
    segmenter_module.segment(
        wrap(Global.shared_image, Global.shm_image),
        wrap(Global.shared_segmentation, Global.shm_segmentation),
        config[segmenter]["default_parameters"],
    )
    return cast(napari.types.LabelsData, Global.shared_segmentation)

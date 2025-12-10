from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from magicgui import magic_factory
from napari.types import LayerDataTuple

from wetlands.environment_manager import (  # type: ignore
    EnvironmentManager,  # type: ignore
)
from wetlands.ndarray import NDArray, update_ndarray

if TYPE_CHECKING:
    import napari
    import napari.types

SEGMENTERS_PATH = Path(__file__).resolve().parent

environemnt_manager = EnvironmentManager(debug=True)
env_cellpose = environemnt_manager.create(
    "Cellpose",
    {
        "python": "3.10",
        "pip": ["wetlands==0.4.4"],
        "conda": ["cellpose==3.1.0"],
    },
)
env_stardist = environemnt_manager.create(
    "StarDist",
    {
        "python": "3.10",
        "pip": [
            "wetlands==0.4.4",
            "tensorflow==2.16.1",
            "csbdeep==0.8.1",
            "stardist==0.9.1",
        ],
    },
)
env_sam = environemnt_manager.create(
    "SAM",
    {
        "python": "3.10",
        "pip": ["wetlands==0.4.4", "sam2==1.1.0", "huggingface_hub==0.29.2"],
    },
)
for env in [env_cellpose, env_stardist, env_sam]:
    env.launch()

shared_image: NDArray | None = None
shared_segmentation: NDArray | None = None


def update_shared_memory(image: "napari.types.ImageData"):
    global shared_image, shared_segmentation
    if image is None:
        return
    shared_image = update_ndarray(image, shared_image)
    shared_segmentation = update_ndarray(
        np.zeros(image.shape[:-1], "uint8"), shared_segmentation
    )


@magic_factory(model_type={"choices": ["cyto3", "cyto2", "nuclei"]})
def cellpose(
    image: "napari.types.ImageData",
    model_type="cyto3",
    use_gpu: bool = True,
    diameter: float = 30.0,
) -> LayerDataTuple:
    global shared_image, shared_segmentation
    update_shared_memory(image)
    env_cellpose.execute(
        SEGMENTERS_PATH / "_cellpose.py",
        "segment",
        (
            shared_image,
            shared_segmentation,
            {
                "model_type": model_type,
                "use_gpu": use_gpu,
                "diameter": diameter,
                "channels": [0, 0],
            },
        ),
    )
    if shared_segmentation is None:
        raise Exception("The shared segmentation is not initialized.")
    return cast(
        LayerDataTuple,
        (
            shared_segmentation.array,
            {"name": "Cellpose segmentation"},
            "labels",
        ),
    )


@magic_factory(
    model_name={"choices": ["2D_versatile_fluo", "2D_paper_dsb2018"]}
)
def stardist(
    image: "napari.types.ImageData", model_name="2D_versatile_fluo"
) -> LayerDataTuple:
    global shared_image, shared_segmentation
    update_shared_memory(image)
    env_stardist.execute(
        SEGMENTERS_PATH / "_stardist.py",
        "segment",
        (
            shared_image,
            shared_segmentation,
            {"model_name": model_name},
        ),
    )
    if shared_segmentation is None:
        raise Exception("The shared segmentation is not initialized.")

    return cast(
        LayerDataTuple,
        (
            shared_segmentation.array,
            {"name": "Stardist segmentation"},
            "labels",
        ),
    )


@magic_factory()
def sam(
    image: "napari.types.ImageData",
    use_gpu: bool = True,
    points_per_side: int = 8,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
) -> LayerDataTuple:
    global shared_image, shared_segmentation
    update_shared_memory(image)
    env_sam.execute(
        SEGMENTERS_PATH / "_sam.py",
        "segment",
        (
            shared_image,
            shared_segmentation,
            {
                "use_gpu": use_gpu,
                "points_per_side": points_per_side,
                "pred_iou_thresh": pred_iou_thresh,
                "stability_score_thresh": stability_score_thresh,
            },
        ),
    )
    if shared_segmentation is None:
        raise Exception("The shared segmentation is not initialized.")
    return cast(
        LayerDataTuple,
        (shared_segmentation.array, {"name": "SAM segmentation"}, "labels"),
    )


@magic_factory(
    call_button="Exit plugin",
)
def exit_button():
    exit_environments()


def exit_environments():
    if shared_image is not None:
        shared_image.dispose()
    if shared_segmentation is not None:
        shared_segmentation.dispose()
    environemnt_manager.exit()


# I need something like this
# segmenter_widget.closed.connect(exit_environments)
